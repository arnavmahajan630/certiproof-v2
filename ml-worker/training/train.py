import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import sys
from pathlib import Path

ML_WORKER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_WORKER_ROOT))

from app.rcajx.model import RCAJ_X

DATA_DIR = ML_WORKER_ROOT / "data"
RESULTS_DIR = ML_WORKER_ROOT / "results"
CHECKPOINT_DIR = ML_WORKER_ROOT / "app" / "models" / "rcajx"


def _device() -> torch.device:
    """CUDA > MPS > CPU. Training-only concern -- inference (app/rcajx/) stays
    CPU-only by design (matches the Docker/API-serving image, no GPU dependency
    for the served app). Set RCAJX_TRAIN_DEVICE=cpu to force CPU even with a GPU
    present (e.g. to reproduce a run bit-for-bit against a CPU-only machine)."""
    forced = os.environ.get("RCAJX_TRAIN_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_device(ex: dict, device: torch.device) -> dict:
    return {
        **ex,
        "R": ex["R"].to(device),
        "A": ex["A"].to(device),
        "negation_flags": ex["negation_flags"].to(device),
        "max_marks": ex["max_marks"].to(device),
    }


def evaluate_loss(model, examples, loss_fn, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for ex in examples:
            out = model(ex["R"], ex["A"], ex["negation_flags"], ex["max_marks"])
            target = torch.tensor(list(ex["human_scores"].values()), dtype=torch.float32, device=device)
            total += loss_fn(out["per_criterion_scores"], target).item()
    return total / len(examples) if examples else 0.0

def train_one_config(train_examples, val_examples, n_heads, d_k, lr, weight_decay, epochs=100, device=None):
    device = device or torch.device("cpu")
    model = RCAJ_X(n_heads=n_heads, d_k=d_k, d_v=d_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for ex in train_examples:
            optimizer.zero_grad()
            out = model(ex["R"], ex["A"], ex["negation_flags"], ex["max_marks"])
            target = torch.tensor(list(ex["human_scores"].values()), dtype=torch.float32, device=device)
            loss = loss_fn(out["per_criterion_scores"], target)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_examples, loss_fn, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Deep copy, moved to CPU -- the saved checkpoint must be loadable on
            # any machine (incl. the ml-worker API, which runs CPU-only), regardless
            # of what device this training run used.
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_loss

def compute_mean_baseline(train_examples, val_examples, loss_fn):
    # Compute mean training score per criterion across all examples?
    # Actually, scores are relative to questions/criteria.
    # Predict the mean training score for each specific criterion.
    from collections import defaultdict
    criterion_sums = defaultdict(float)
    criterion_counts = defaultdict(int)
    
    for ex in train_examples:
        for crit_id, score in ex["human_scores"].items():
            criterion_sums[crit_id] += score
            criterion_counts[crit_id] += 1
            
    criterion_means = {k: v / criterion_counts[k] for k, v in criterion_sums.items()}
    
    # Calculate baseline loss on val set
    total_loss = 0.0
    for ex in val_examples:
        preds = []
        targets = []
        for crit_id, score in ex["human_scores"].items():
            preds.append(criterion_means.get(crit_id, 0.0))
            targets.append(score)
        
        preds_tensor = torch.tensor(preds, dtype=torch.float32)
        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        total_loss += loss_fn(preds_tensor, targets_tensor).item()
        
    return total_loss / len(val_examples) if val_examples else 0.0

def run_training():
    device = _device()
    print(f"Training device: {device}")

    print("Loading training data...")
    all_train_examples = torch.load(DATA_DIR / "train_embedded.pt", weights_only=False)

    for ex in all_train_examples:
        targets = torch.tensor(list(ex["human_scores"].values()), dtype=torch.float32)
        assert torch.all(targets >= 0) and torch.all(targets <= ex["max_marks"]), \
            f"human_scores out of [0, max_marks] range for {ex['answer_id']}"

    # Needs at least enough examples to stratify
    try:
        train_examples, val_examples = train_test_split(
            all_train_examples, test_size=0.2, random_state=42,
            stratify=[ex["question_id"] for ex in all_train_examples]
        )
    except ValueError:
        # If not enough examples to stratify (e.g. dummy data), just split
        train_examples, val_examples = train_test_split(
            all_train_examples, test_size=0.2, random_state=42
        )

    if device.type != "cpu":
        train_examples = [_to_device(ex, device) for ex in train_examples]
        val_examples = [_to_device(ex, device) for ex in val_examples]

    grid = [
        {"n_heads": h, "d_k": dk, "lr": lr, "weight_decay": wd}
        for h in [2, 4]
        for dk in [64]
        for lr in [5e-3, 1e-3]
        for wd in [1e-4]
    ]

    results = []
    overall_best_val_loss = float("inf")
    overall_best_state = None
    overall_best_config = None

    print(f"Running ablation over {len(grid)} configurations...")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    for i, config in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] Training config: {config}")
        best_state, best_val_loss = train_one_config(train_examples, val_examples, **config, device=device)
        
        config_result = config.copy()
        config_result["best_val_loss"] = best_val_loss
        results.append(config_result)
        
        if best_val_loss < overall_best_val_loss:
            overall_best_val_loss = best_val_loss
            overall_best_state = best_state
            overall_best_config = config
            
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)

    # Save best model
    torch.save({
        "state_dict": overall_best_state,
        "config": overall_best_config,
        "val_loss": overall_best_val_loss,
    }, CHECKPOINT_DIR / "rcaj_x_best.pt")
    
    # Compute baseline
    loss_fn = nn.MSELoss()
    baseline_loss = compute_mean_baseline(train_examples, val_examples, loss_fn)
    
    print("\nTraining Complete.")
    print(f"Best Config: {overall_best_config}")
    print(f"Best Validation Loss: {overall_best_val_loss:.4f}")
    print(f"Mean-Predict Baseline Loss: {baseline_loss:.4f}")

if __name__ == "__main__":
    run_training()
