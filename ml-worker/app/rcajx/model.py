import torch
import torch.nn as nn

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, d_k: int = 64, d_v: int = 64):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_k
        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_v, bias=False)
        self.W_o = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, R: torch.Tensor, A: torch.Tensor, key_mask: torch.Tensor = None):
        # R: (n_criteria, d_model), A: (n_chunks, d_model)
        # key_mask (optional): (n_chunks,) float, 1.0 = real chunk, 0.0 = padding.
        # Padded keys get their pre-softmax score driven to -inf so they receive ~0
        # attention weight and don't distort the softmax over real chunks — needed
        # once R/A are padded to a fixed (MAX_CRITERIA, MAX_CHUNKS) shape for ONNX/EZKL
        # static-shape export (see padded_model.py).
        n_c, n_a = R.shape[0], A.shape[0]
        Q = self.W_q(R).view(n_c, self.n_heads, self.d_k).transpose(0, 1)   # (h, n_c, d_k)
        K = self.W_k(A).view(n_a, self.n_heads, self.d_k).transpose(0, 1)   # (h, n_a, d_k)
        V = self.W_v(A).view(n_a, self.n_heads, self.d_k).transpose(0, 1)   # (h, n_a, d_v)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)   # (h, n_c, n_a)
        if key_mask is not None:
            # A large-but-bounded negative constant, not torch.finfo().min (~-3.4e38):
            # attention scores here are O(10) at most (bounded embeddings through a
            # Linear layer, scaled by 1/sqrt(d_k)), so -1e4 is already many orders of
            # magnitude past anything softmax needs to zero a slot out, and stays
            # inside a range EZKL's fixed-point quantization can represent — the
            # true float min blows up the circuit's required scale/lookup range.
            MASK_NEG = -1e4
            additive_mask = (1.0 - key_mask) * MASK_NEG                     # (n_a,), 0 for real, -1e4 for padded
            scores = scores + additive_mask.view(1, 1, n_a)
        weights = torch.softmax(scores, dim=-1)                            # (h, n_c, n_a), sums to 1 per criterion per head
        context = torch.matmul(weights, V)                                 # (h, n_c, d_v)

        context = context.transpose(0, 1).reshape(n_c, -1)                 # (n_c, h*d_v)
        out = self.W_o(context)                                            # (n_c, d_model)
        return out, weights

class ScoringHead(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, hidden: int = 32):
        super().__init__()
        # input = context (d_model) + per-head spread (n_heads) + negation flag (1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model + n_heads + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, context: torch.Tensor, weights: torch.Tensor, negation_flags: torch.Tensor, max_marks: torch.Tensor, key_mask: torch.Tensor = None):
        # weights: (h, n_c, n_a) -> spread per head per criterion.
        # key_mask (optional): (n_a,), 1.0 = real chunk, 0.0 = padding. Padded
        # chunks carry ~0 attention weight (softmax already masked them out in
        # MultiHeadCrossAttention) but a plain .mean(dim=-1) would still average
        # over all n_a slots including the padded ones, diluting mean_w and
        # inflating spread whenever MAX_CHUNKS >> real chunk count. Masked mean
        # (sum over real chunks / count of real chunks) keeps spread identical
        # to the unpadded computation regardless of how much padding is present.
        max_w = weights.max(dim=-1).values      # (h, n_c)
        if key_mask is not None:
            n_real = key_mask.sum().clamp(min=1.0)
            mean_w = weights.sum(dim=-1) / n_real   # (h, n_c)
        else:
            mean_w = weights.mean(dim=-1)            # (h, n_c)
        spread = (max_w - mean_w).transpose(0, 1)  # (n_c, h)
        x = torch.cat([context, spread, negation_flags.unsqueeze(-1)], dim=-1)
        raw = self.mlp(x).squeeze(-1)
        # Bound to [0, max_marks] via sigmoid scaling (constrains the function's range,
        # not just a post-hoc clamp) — see rcaj-x-hardening-plan/01 for why clamping alone isn't enough.
        bounded_score = torch.sigmoid(raw) * max_marks
        return bounded_score, spread     # return spread too, needed for benchmark analysis

class RCAJ_X(nn.Module):
    def __init__(self, d_model: int = 384, n_heads: int = 4, d_k: int = 64, d_v: int = 64, hidden: int = 32):
        super().__init__()
        self.attn = MultiHeadCrossAttention(d_model, n_heads, d_k, d_v)
        self.score_head = ScoringHead(d_model, n_heads, hidden)

    def forward(self, R: torch.Tensor, A: torch.Tensor, negation_flags: torch.Tensor, max_marks: torch.Tensor, criterion_weights: torch.Tensor = None, key_mask: torch.Tensor = None):
        context, weights = self.attn(R, A, key_mask=key_mask)
        per_criterion_scores, spread = self.score_head(context, weights, negation_flags, max_marks, key_mask=key_mask)
        final_score = None
        if criterion_weights is not None:
            final_score = (per_criterion_scores * criterion_weights).sum()
        return {
            "per_criterion_scores": per_criterion_scores,
            "final_score": final_score,
            "attn_weights": weights,
            "spread": spread,
        }
