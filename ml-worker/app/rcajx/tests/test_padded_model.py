import torch

from app.rcajx.model import RCAJ_X
from app.rcajx.padded_model import MAX_CHUNKS, MAX_CRITERIA, PaddedRCAJX, pad_inputs


def _random_example(n_criteria, n_chunks, seed=0):
    torch.manual_seed(seed)
    R = torch.randn(n_criteria, 384)
    A = torch.randn(n_chunks, 384)
    neg = torch.randint(0, 2, (n_criteria,)).float()
    max_marks = torch.rand(n_criteria) * 3 + 1
    weights = torch.rand(n_criteria)
    return R, A, neg, max_marks, weights


def test_padded_matches_unpadded_for_various_shapes():
    model = RCAJ_X(n_heads=2, d_k=64, d_v=64)
    model.eval()
    wrapper = PaddedRCAJX(model)

    for n_criteria, n_chunks in [(1, 1), (3, 5), (10, 24), (7, 2)]:
        R, A, neg, max_marks, weights = _random_example(n_criteria, n_chunks)
        with torch.no_grad():
            ref = model(R, A, neg, max_marks, criterion_weights=weights)
        padded = pad_inputs(R, A, neg, max_marks, criterion_weights=weights)
        with torch.no_grad():
            per_crit, final_score, attn_w, spread = wrapper(
                padded["R_padded"], padded["A_padded"], padded["negation_flags_padded"],
                padded["max_marks_padded"], padded["chunk_mask"], padded["criterion_weights_padded"],
            )
        n_c = padded["n_criteria"]
        assert torch.allclose(ref["per_criterion_scores"], per_crit[:n_c], atol=1e-5)
        assert torch.allclose(ref["spread"], spread[:n_c], atol=1e-5)
        assert torch.allclose(ref["final_score"], final_score, atol=1e-5)


def test_padded_chunks_get_zero_attention():
    model = RCAJ_X(n_heads=2, d_k=64, d_v=64)
    model.eval()
    wrapper = PaddedRCAJX(model)
    R, A, neg, max_marks, weights = _random_example(3, 5)
    padded = pad_inputs(R, A, neg, max_marks, criterion_weights=weights)
    with torch.no_grad():
        _, _, attn_w, _ = wrapper(
            padded["R_padded"], padded["A_padded"], padded["negation_flags_padded"],
            padded["max_marks_padded"], padded["chunk_mask"], padded["criterion_weights_padded"],
        )
    n_c, n_a = padded["n_criteria"], padded["n_chunks"]
    assert attn_w[:, :n_c, n_a:].abs().max().item() == 0.0


def test_padded_output_shapes_always_static():
    model = RCAJ_X(n_heads=2, d_k=64, d_v=64)
    model.eval()
    wrapper = PaddedRCAJX(model)
    R, A, neg, max_marks, weights = _random_example(2, 3)
    padded = pad_inputs(R, A, neg, max_marks, criterion_weights=weights)
    with torch.no_grad():
        per_crit, final_score, attn_w, spread = wrapper(
            padded["R_padded"], padded["A_padded"], padded["negation_flags_padded"],
            padded["max_marks_padded"], padded["chunk_mask"], padded["criterion_weights_padded"],
        )
    assert per_crit.shape == (MAX_CRITERIA,)
    assert spread.shape == (MAX_CRITERIA, 2)
    assert attn_w.shape == (2, MAX_CRITERIA, MAX_CHUNKS)
    assert final_score.dim() == 0
