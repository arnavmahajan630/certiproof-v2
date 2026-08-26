import torch

from app.rcajx.model import RCAJ_X
from app.rcajx.guardrails import input_sanity_check
from app.rcajx.preprocessing import embed_criteria, embed_answer, negation_mismatch_flag

CRITERIA_TEXTS = ["Explains the process clearly", "Uses correct terminology"]
MAX_MARKS = torch.tensor([2.0, 2.0])

ADVERSARIAL_INPUTS = [
    "Yes.",                                                    # single-word
    "Explain the process clearly using correct terminology.",  # restates the question
    "idk man its like the thing that does the stuff u kno",    # register mismatch (informal)
    "The French Revolution began in 1789 due to fiscal crisis and social inequality.",  # topic mismatch
    "",                                                         # empty
    "   ",                                                      # whitespace-only
]


def _grade(model, answer_text):
    glossary = {}
    R = embed_criteria(CRITERIA_TEXTS)
    ans = embed_answer(answer_text if answer_text.strip() else "placeholder", glossary)
    A, chunks = ans["A"], ans["chunks"]
    neg_flags = torch.tensor(
        [negation_mismatch_flag(c, chunks[0]) for c in CRITERIA_TEXTS], dtype=torch.float32
    )
    return model(R, A, neg_flags, MAX_MARKS)


def test_adversarial_inputs_bounded_and_flagged():
    torch.manual_seed(0)
    model = RCAJ_X(n_heads=4)
    for text in ADVERSARIAL_INPUTS:
        sanity_issues = input_sanity_check(text)
        out = _grade(model, text)
        scores = out["per_criterion_scores"]
        assert torch.all(scores >= 0) and torch.all(scores <= MAX_MARKS), f"Unbounded score for input: {text!r}"
        mean_spread = out["spread"].mean().item()
        model_flagged_low_confidence = mean_spread < 0.4
        assert sanity_issues or model_flagged_low_confidence, (
            f"No guardrail fired for adversarial input {text!r} — "
            f"sanity_issues={sanity_issues}, mean_spread={mean_spread}"
        )
