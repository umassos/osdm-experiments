import torch
import torch.nn as nn
import torch.nn.functional as F


class MonotoneHead(nn.Module):
    """
        Produces a length-K non-increasing, concave vector y in [low, high] with a
        softplus-based construction that avoids gate saturation.

    Construction:
        - Parameterize first-difference magnitudes a_i (i=0..K-2) as a non-decreasing sequence:
                e = softplus(l[:K-1]) + eps,  a_unnorm = cumsum(e)
            so a_unnorm is non-decreasing.
        - Normalize w = a_unnorm / sum(a_unnorm) to distribute total drop (top-low) across steps.
        - Cumulative fractions C = [0, cumsum(w)], length K, so C[0]=0, C[K-1]=1.
    - A top parameter g_top in R sets the starting level via a temperature-scaled sigmoid:
        top = low + (high - low) * sigmoid(g_top / temp).
        - Define y_i = top - (top - low) * C[i]. Then y_0 = top and y_{K-1} = low.

        Properties:
        - y is non-increasing and concave (second differences ≤ 0). Linear is a special case
            when a_unnorm is constant (e has only its first entry > 0).

    Notes:
    - dec_gate is accepted for backward compatibility but ignored.
    - temp > 1 softens the sigmoid to mitigate saturation.
    """

    def __init__(self, K: int, temp: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.K = K
        self.temp = float(temp)
        self.eps = float(eps)

    def forward(
        self,
        logits: torch.Tensor,
        top_gate: torch.Tensor,
        dec_gate: torch.Tensor,  # unused; kept for interface compatibility
        low: float,
        high: float,
    ) -> torch.Tensor:
        # logits: (..., K), top_gate: (..., 1)
        K = self.K
        # Handle degenerate K=1 case
        if K == 1:
            top_frac = torch.sigmoid(top_gate / self.temp)
            top = low + (high - low) * top_frac
            return top

        # First-difference magnitudes (length K-1), enforced non-decreasing by cumulative sum
        e = F.softplus(logits[..., : K - 1]) + self.eps            # (..., K-1)
        a_unnorm = torch.cumsum(e, dim=-1)                         # (..., K-1), non-decreasing
        sum_a = torch.clamp(a_unnorm.sum(dim=-1, keepdim=True), min=self.eps)
        w = a_unnorm / sum_a                                       # (..., K-1), sum to 1
        csum_w = torch.cumsum(w, dim=-1)                           # (..., K-1), last=1
        zeros = torch.zeros_like(csum_w[..., :1])
        C = torch.cat([zeros, csum_w], dim=-1)                     # (..., K), C[0]=0, C[-1]=1

        # Top within [low, high] using softened sigmoid
        top_frac = torch.sigmoid(top_gate / self.temp)  # (..., 1)
        top = low + (high - low) * top_frac

        # Monotone non-increasing thresholds
        y = top - (top - low) * C
        return y


class ThresholdPredictor(nn.Module):
    """
    Simple MLP that maps a driver/context feature vector to three K-length
    threshold vectors: base (y), flex purchase (y_p), and flex delivery (y_d).

    The outputs are guaranteed monotone non-increasing and clipped to [p_min, p_max]
    via MonotoneHead.
    """

    def __init__(self, input_dim: int, K: int, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Three heads, each outputs:
        # - K logits (softmax weights)
        # - 1 top gate (controls y[0] within [low, high])
        # - 1 decrease gate (controls total drop toward low)
        self.base_head_logits = nn.Linear(prev, K)
        self.base_head_top = nn.Linear(prev, 1)
        self.base_head_dec = nn.Linear(prev, 1)

        self.flex_p_head_logits = nn.Linear(prev, K)
        self.flex_p_head_top = nn.Linear(prev, 1)
        self.flex_p_head_dec = nn.Linear(prev, 1)

        self.flex_d_head_logits = nn.Linear(prev, K)
        self.flex_d_head_top = nn.Linear(prev, 1)
        self.flex_d_head_dec = nn.Linear(prev, 1)

        # Softplus-based monotone head (avoids gate saturation)
        self.mono = MonotoneHead(K)

        # Initialize gate biases to encourage reasonable starting thresholds
        # top_frac ≈ 0.6 (slightly above mid-range); dec gate exists but unused by the head
        init_top_bias = torch.tensor(0.4055)  # logit(0.6)
        init_dec_bias = torch.tensor(0.0)     # dec gate unused; keep neutral
        for layer in [self.base_head_top, self.flex_p_head_top, self.flex_d_head_top]:
            nn.init.constant_(layer.bias, float(init_top_bias))
            nn.init.zeros_(layer.weight)
        for layer in [self.base_head_dec, self.flex_p_head_dec, self.flex_d_head_dec]:
            nn.init.constant_(layer.bias, float(init_dec_bias))
            nn.init.zeros_(layer.weight)

    def forward(self, x: torch.Tensor, p_min: float, p_max: float):
        """
        x: (B, F) feature tensor or (F,) for single example.
        Returns: y_base, y_flex_p, y_flex_d each shape (B, K) or (K,) if input was 1D.
        """
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True

        h = self.trunk(x)
        lb, ub = float(p_min), float(p_max)

        y_base = self.mono(
            self.base_head_logits(h), self.base_head_top(h), self.base_head_dec(h), lb, ub
        )
        y_fp = self.mono(
            self.flex_p_head_logits(h), self.flex_p_head_top(h), self.flex_p_head_dec(h), lb, ub
        )
        y_fd = self.mono(
            self.flex_d_head_logits(h), self.flex_d_head_top(h), self.flex_d_head_dec(h), lb, ub
        )

        # fixed offset (to be added to predicted vector) (K-1)e-5 at location 0, (K-2)e-5 at location 1, 1e-5 at location -2, ..., 0 at location -1
        offsets = torch.arange(self.mono.K - 1, -1, -1, dtype=y_base.dtype, device=y_base.device) * 1e-5
        y_base = y_base + offsets
        y_fp = y_fp + offsets
        y_fd = y_fd + offsets

        if squeeze_out:
            return y_base[0], y_fp[0], y_fd[0]
        return y_base, y_fp, y_fd


__all__ = ["ThresholdPredictor", "MonotoneHead"]
