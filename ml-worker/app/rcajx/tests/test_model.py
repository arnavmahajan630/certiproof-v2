import torch
import pytest

from app.rcajx.model import RCAJ_X

def test_softmax_normalization():
    model = RCAJ_X(n_heads=4)
    R, A = torch.randn(3, 384), torch.randn(6, 384)
    neg = torch.zeros(3)
    max_marks = torch.tensor([2.0, 2.0, 1.0])
    out = model(R, A, neg, max_marks)
    assert torch.allclose(out["attn_weights"].sum(dim=-1), torch.ones(4, 3), atol=1e-5)

def test_output_shapes():
    model = RCAJ_X(n_heads=4)
    R, A = torch.randn(5, 384), torch.randn(10, 384)
    neg = torch.zeros(5)
    max_marks = torch.tensor([2.0, 2.0, 1.0, 3.0, 2.0])
    weights = torch.ones(5)
    out = model(R, A, neg, max_marks, weights)
    assert out["per_criterion_scores"].shape == (5,)
    assert out["final_score"].dim() == 0

def test_head_count_configurable():
    for h in [2, 4, 6]:
        model = RCAJ_X(n_heads=h)
        R, A = torch.randn(3, 384), torch.randn(4, 384)
        max_marks = torch.tensor([2.0, 2.0, 1.0])
        out = model(R, A, torch.zeros(3), max_marks)
        assert out["attn_weights"].shape[0] == h

def test_scores_bounded_by_max_marks():
    torch.manual_seed(0)
    model = RCAJ_X(n_heads=4)
    for _ in range(20):
        n_c, n_a = 5, 8
        R, A = torch.randn(n_c, 384) * 3, torch.randn(n_a, 384) * 3  # exaggerated magnitudes
        neg = torch.randint(0, 2, (n_c,)).float()
        max_marks = torch.tensor([2.0, 1.0, 3.0, 2.0, 1.0])
        out = model(R, A, neg, max_marks)
        assert torch.all(out["per_criterion_scores"] >= 0)
        assert torch.all(out["per_criterion_scores"] <= max_marks)
