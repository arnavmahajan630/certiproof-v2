"""Fixed-shape padding/masking wrapper around RCAJ_X.

RCAj-x-mini-v1's model operates on variable-length (n_criteria, n_chunks) tensors
with no padding at all. CertiProof's EZKL circuit needs static tensor shapes (and
already assumes a MAX_CRITERIA=10 rubric-size cap, matching the old Zone1/Zone2
contract). This module is net-new — it does not exist in RCAj-x-mini-v1 — and is
the thing that gets ONNX-exported / compiled into the circuit, never RCAJ_X
directly.

MAX_CRITERIA=10 matches CertiProof's existing rubric limit. MAX_CHUNKS=24 is a
generous cap on the number of answer-sentence chunks (picked to comfortably cover
the >500-word guardrail limit in guardrails.py at typical sentence length; answers
producing more than 24 chunks are truncated to the first 24 — same "abnormally
long" territory input_sanity_check already flags).

Padding scheme:
- R padded to (MAX_CRITERIA, d_model) with zero rows — harmless: each criterion's
  attention output only depends on its own query row, so padded criteria rows
  never influence real ones. Callers must truncate outputs back to n_criteria.
- A padded to (MAX_CHUNKS, d_model) with zero rows, PLUS an explicit chunk_mask
  (1.0 = real chunk, 0.0 = padding) passed into MultiHeadCrossAttention so padded
  chunks get their pre-softmax attention score driven to -inf — without this,
  padded (zero) chunks would receive real (nonzero) attention weight and dilute
  the softmax over genuine chunks.
- negation_flags / max_marks padded to MAX_CRITERIA with zeros.
- criterion_weights (if used for final_score aggregation), padded to MAX_CRITERIA
  with zeros so padded criteria never contribute to the weighted sum.
"""
import torch
import torch.nn as nn

from .model import RCAJ_X

MAX_CRITERIA = 10
MAX_CHUNKS = 24
D_MODEL = 384


def pad_rows(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad/truncate a (n, d) tensor to (target_len, d) with zero rows."""
    n = x.shape[0]
    if n >= target_len:
        return x[:target_len]
    pad = torch.zeros(target_len - n, x.shape[1], dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def pad_vec(x: torch.Tensor, target_len: int) -> torch.Tensor:
    n = x.shape[0]
    if n >= target_len:
        return x[:target_len]
    pad = torch.zeros(target_len - n, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def build_chunk_mask(n_chunks: int, max_chunks: int = MAX_CHUNKS) -> torch.Tensor:
    real = min(n_chunks, max_chunks)
    mask = torch.zeros(max_chunks, dtype=torch.float32)
    mask[:real] = 1.0
    return mask


def pad_inputs(
    R: torch.Tensor,
    A: torch.Tensor,
    negation_flags: torch.Tensor,
    max_marks: torch.Tensor,
    criterion_weights: torch.Tensor = None,
) -> dict:
    """Pad a raw (variable-length) example's tensors to the fixed shapes the
    circuit expects. Returns a dict of padded tensors plus n_criteria/n_chunks
    (the real, pre-padding counts) so callers can truncate outputs afterward."""
    n_criteria = min(R.shape[0], MAX_CRITERIA)
    n_chunks = min(A.shape[0], MAX_CHUNKS)

    out = {
        "R_padded": pad_rows(R, MAX_CRITERIA),
        "A_padded": pad_rows(A, MAX_CHUNKS),
        "negation_flags_padded": pad_vec(negation_flags, MAX_CRITERIA),
        "max_marks_padded": pad_vec(max_marks, MAX_CRITERIA),
        "chunk_mask": build_chunk_mask(A.shape[0]),
        "n_criteria": n_criteria,
        "n_chunks": n_chunks,
    }
    if criterion_weights is not None:
        out["criterion_weights_padded"] = pad_vec(criterion_weights, MAX_CRITERIA)
    return out


class PaddedRCAJX(nn.Module):
    """Wraps RCAJ_X for a fixed (MAX_CRITERIA, MAX_CHUNKS) input shape — the
    module actually traced for ONNX export / EZKL compilation. All inputs must
    already be padded (see pad_inputs above); this module does no padding itself
    so its graph has purely static shapes."""

    def __init__(self, rcaj_x: RCAJ_X):
        super().__init__()
        self.model = rcaj_x

    def forward(
        self,
        R_padded: torch.Tensor,
        A_padded: torch.Tensor,
        negation_flags_padded: torch.Tensor,
        max_marks_padded: torch.Tensor,
        chunk_mask: torch.Tensor,
        criterion_weights_padded: torch.Tensor,
    ):
        # criterion_weights_padded is required (not Optional) here: ONNX export
        # needs a fixed, non-None output set, and RCAJ_X.forward returns
        # final_score=None whenever criterion_weights is None. Pass an
        # all-ones-then-zero-padded weight vector if you just want per-criterion
        # scores and don't care about final_score.
        out = self.model(
            R_padded,
            A_padded,
            negation_flags_padded,
            max_marks_padded,
            criterion_weights=criterion_weights_padded,
            key_mask=chunk_mask,
        )
        return out["per_criterion_scores"], out["final_score"], out["attn_weights"], out["spread"]
