import torch
import torch.nn as nn
import torch.nn.functional as F
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp


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

    def __init__(self, 
                input_dim: int, 
                K: int, 
                hidden_dims=(64, 64),
                p_min: float = 1.0,
                p_max: float = 200.0,
                gamma: float = 10.0,
                delta: float = 5.0,
                c_delivery: float = 0.2,
                eps_delivery: float = 0.05,
                T: int = 48,
                beta: float = 30,
                use_robust_projection: bool = False,
                concavity_eps: float = 1e-6,
                enforce_concavity_post: bool = True,
                 ):
        super().__init__()
        self.const = {
            "p_min": float(p_min),
            "p_max": float(p_max),
            "gamma": float(gamma),
            "delta": float(delta),
            "c": float(c_delivery),
            "eps": float(eps_delivery),
            "T": int(T),
            "beta": float(beta),
        }
        self.use_robust_projection = use_robust_projection
        self.concavity_eps = float(concavity_eps)
        self.enforce_concavity_post = bool(enforce_concavity_post)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.trunk = nn.Sequential(*layers)

        # Three heads, each outputs:
        # - K logits (shape/concavity via cumulative softplus)
        # - 1 top gate (controls H relative to L, i.e., y[0] ∈ [L, high])
        # - 1 bottom gate (controls L ∈ [low, high])
        self.K = K
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
        # bottom_frac ≈ 0.4, top_frac ≈ 0.6 (so initial H > L, both near mid-range)
        init_top_bias = torch.tensor(0.4055)   # logit(0.6)
        init_dec_bias = torch.tensor(-0.4055)  # logit(0.4)
        for layer in [self.base_head_top, self.flex_p_head_top, self.flex_d_head_top]:
            nn.init.constant_(layer.bias, float(init_top_bias))
            nn.init.zeros_(layer.weight)
        for layer in [self.base_head_dec, self.flex_p_head_dec, self.flex_d_head_dec]:
            nn.init.constant_(layer.bias, float(init_dec_bias))
            nn.init.zeros_(layer.weight)

        # ----------------- CVXPY projection layers (optional) -----------------
        if self.use_robust_projection:
            self.base_proj_layer = self._build_base_projection_layer()
            self.flex_proj_layer = self._build_flex_projection_layer()
        else:
            self.base_proj_layer = None
            self.flex_proj_layer = None

    # ---- CVXPY layer builders ----
    def _build_base_projection_layer(self) -> CvxpyLayer:
        K = self.K
        p_min = self.const["p_min"]
        p_max = self.const["p_max"]
        gamma = self.const["gamma"]
        delta = self.const["delta"]
        c = self.const["c"]
        eps = self.const["eps"]
        T = self.const["T"]
        beta = self.const["beta"]

        y_param = cp.Parameter(K)
        y_var = cp.Variable(K)
        # First-difference variables: d[i] = y[i+1] - y[i]
        d = cp.Variable(K - 1) if K > 1 else None

        cons = []
        # Bounds
        for i in range(K):
            cons += [y_var[i] >= p_min, y_var[i] <= p_max]
        # Monotone non-increasing y (equivalently, forward differences <= 0)
        for i in range(1, K):
            cons += [y_var[i] <= y_var[i - 1]]
        # Enforce concavity via first differences being negative and strictly nonincreasing in value
        # That is: d[i] = y[i+1] - y[i] <= -eps and d[i] <= d[i-1] - eps (so |d[i]| increases)
        if K > 1:
            for i in range(K - 1):
                cons += [d[i] == y_var[i + 1] - y_var[i], d[i] <= -self.concavity_eps]
        if K > 2:
            for i in range(1, K - 1):
                cons += [d[i] <= d[i - 1] - self.concavity_eps]
        # Tail upper bound (original code used <=; add equality if you want strict pin)
        cons += [y_var[K - 1] <= p_min + 2.0 * gamma]

        # Robustness inequalities (same discretization as project_y_robust)
        for j in range(0, K + 1):
            w = j / K
            if j == 0:
                integral = 0
            else:
                integral = (1.0 / K) * cp.sum(y_var[:j])
            lhs = (
                integral
                + (1 - w) * (p_max + 2 * gamma)
                + p_max * (c + eps)
                + 2 * delta
                - c * w * p_min
            )
            psi_w = y_var[j - 1] if j >= 1 else y_var[0]
            rhs = beta * (psi_w - 2 * gamma + eps * p_max + (2 * delta + 2 * gamma) / T)
            cons += [lhs <= rhs]

        obj = cp.Minimize(cp.sum_squares(y_var - y_param))
        prob = cp.Problem(obj, cons)
        return CvxpyLayer(prob, parameters=[y_param], variables=[y_var])

    def _build_flex_projection_layer(self) -> CvxpyLayer:
        K = self.K
        p_min = self.const["p_min"]
        p_max = self.const["p_max"]
        gamma = self.const["gamma"]
        delta = self.const["delta"]
        c = self.const["c"]
        eps = self.const["eps"]
        T = self.const["T"]
        beta = self.const["beta"]

        phi_param = cp.Parameter(K)  # flex purchase proto
        psi_param = cp.Parameter(K)  # flex delivery proto
        phi = cp.Variable(K)
        psi = cp.Variable(K)
        # First-difference variables for concavity and monotonicity
        d_phi = cp.Variable(K - 1) if K > 1 else None
        d_psi = cp.Variable(K - 1) if K > 1 else None

        cons = []
        # Bounds + monotonic
        for i in range(K):
            cons += [phi[i] >= p_min, phi[i] <= p_max]
            cons += [psi[i] >= p_min, psi[i] <= p_max]
        for i in range(1, K):
            cons += [phi[i] <= phi[i - 1]]
            cons += [psi[i] <= psi[i - 1]]
        # Concavity via first differences negative and strictly nonincreasing (|diff| strictly nondecreasing)
        if K > 1:
            for i in range(K - 1):
                cons += [d_phi[i] == phi[i + 1] - phi[i], d_phi[i] <= -self.concavity_eps]
                cons += [d_psi[i] == psi[i + 1] - psi[i], d_psi[i] <= -self.concavity_eps]
        if K > 2:
            for i in range(1, K - 1):
                cons += [d_phi[i] <= d_phi[i - 1] - self.concavity_eps]
                cons += [d_psi[i] <= d_psi[i - 1] - self.concavity_eps]
        # Tail bounds
        cons += [phi[K - 1] <= p_min + 2.0 * gamma]
        cons += [psi[K - 1] <= p_min * (c + eps) + 2.0 * delta]

        # Trapezoid nodes
        phi_nodes = []
        psi_nodes = []
        for j in range(K + 1):
            if j == 0:
                phi_nodes.append(phi[0])
                psi_nodes.append(psi[0])
            elif j == K:
                phi_nodes.append(phi[K - 1])
                psi_nodes.append(psi[K - 1])
            else:
                phi_nodes.append(0.5 * (phi[j - 1] + phi[j]))
                psi_nodes.append(0.5 * (psi[j - 1] + psi[j]))

        # Integrals
        int_phi = [0]
        for j in range(1, K + 1):
            seg = 0.5 * (phi_nodes[j - 1] + phi_nodes[j]) * (1.0 / K)
            int_phi.append(int_phi[-1] + seg)
        int_psi = [0]
        for l in range(1, K + 1):
            seg = 0.5 * (psi_nodes[l - 1] + psi_nodes[l]) * (1.0 / K)
            int_psi.append(int_psi[-1] + seg)

        kappa = gamma + delta
        fac = (1.0 + eps) / (1.0 + c + eps)

        for j in range(0, K + 1):
            w = j / K
            for l in range(0, K + 1):
                v = l / K
                lhs = (
                    int_phi[j]
                    + (1.0 - w) * (p_max + 2.0 * gamma)
                    - c * w * p_min
                    + int_psi[l]
                    + (1.0 - v) * (p_max * (c + eps) + 2.0 * delta)
                )
                rhs = beta * (fac * (phi_nodes[j] + psi_nodes[l] - 2.0 * kappa) + (2.0 * kappa) / T)
                cons += [lhs <= rhs]

        obj = cp.Minimize(cp.sum_squares(phi - phi_param) + cp.sum_squares(psi - psi_param))
        prob = cp.Problem(obj, cons)
        return CvxpyLayer(prob, parameters=[phi_param, psi_param], variables=[phi, psi])

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

        if self.use_robust_projection:
            # print thresholds before projection
            # print("Before projection:")
            # print("Base y:", y_base)
            # print("Flex purchase y_p:", y_fp)
            # print("Flex delivery y_d:", y_fd)
            # Project each batch row independently (cvxpylayers supports batched params)
            (y_base,) = self.base_proj_layer(y_base)
            y_fp, y_fd = self.flex_proj_layer(y_fp, y_fd)
            # print("After projection:")
            # print("Base y:", y_base)
            # print("Flex purchase y_p:", y_fp)
            # print("Flex delivery y_d:", y_fd)

        if squeeze_out:
            return y_base[0], y_fp[0], y_fd[0]
        return y_base, y_fp, y_fd


__all__ = ["ThresholdPredictor", "MonotoneHead"]
