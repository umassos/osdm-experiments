import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer
import torch

"""
PALD CVXPyLayers with integral-aware (per-segment) allocation.
"""

def make_pald_base_layer(K):
    # ---- compile-time constants ----
    taus_full = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    taus = taus_full[:-1]            # locations of hinge knots (exclude last)
    tau0, tauK = float(taus_full[0]), float(taus_full[-1])

    # ---- decision ----
    x = cp.Variable()

    # ---- parameters (layer inputs) ----
    a_t  = cp.Parameter(1,nonneg=True)                 # scalar
    w_const = cp.Parameter(1,nonneg=True)                 # scalar
    p_t     = cp.Parameter(1,nonneg=True)                 # scalar
    eta   = cp.Parameter(1,nonneg=True)                 # scalar
    # IMPORTANT: pass these precomputed from y (outside the graph)
    w_hinge = cp.Parameter(K, nonneg=True)   # hinge weights >= 0
    c1      = cp.Parameter()                 # slope term for F (e.g., -y[0])

    # ---- auxiliaries (nonlinearities live here, no parameters inside atoms) ----
    z = w_const + x
    q = cp.Variable(K)                       # >= (z - tau_j)_+
    r = cp.Variable(K)                       # >= 0.5 * q_j^2
    u1 = cp.Variable()                       # >= |x - a_t|

    constraints = [
        z >= tau0, z <= tauK,
        x >= 0.0, x <= (tauK - w_const),

        q >= z - taus,
        q >= 0,

        cp.square(q) <= 2 * r,   # convex ≤ affine (DPP-safe)
        r >= 0,

        u1 >=  x - a_t,  u1 >= -(x - a_t),
    ]

    # IMPORTANT: no param*param products in the objective
    # Use c1 * x (not c1 * z); the dropped c1*w_const is a parameter-only constant → irrelevant to argmin
    Fz = c1 * x + w_hinge @ r

    objective = cp.Minimize(p_t * x + eta * (u1) + Fz)
    prob = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        prob,
        parameters=[a_t, w_const, p_t, eta, w_hinge, c1],
        variables=[x],
    )

    return layer

def make_pald_flex_purchase_layer(K):
    # ---- compile-time constants ----
    taus_full = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    taus = taus_full[:-1]            # locations of hinge knots (exclude last)
    tau0, tauK = float(taus_full[0]), float(taus_full[-1])

    # ---- decision ----
    x = cp.Variable()

    # ---- parameters (layer inputs) ----
    a_t  = cp.Parameter(1,nonneg=True)                 # scalar
    w_const = cp.Parameter(1,nonneg=True)                 # scalar
    p_t     = cp.Parameter(1,nonneg=True)                 # scalar
    eta   = cp.Parameter(1,nonneg=True)                 # scalar
    # IMPORTANT: pass these precomputed from y (outside the graph)
    w_hinge = cp.Parameter(K, nonneg=True)   # hinge weights >= 0
    c1      = cp.Parameter()                 # slope term for F (e.g., -y[0])

    # ---- auxiliaries (nonlinearities live here, no parameters inside atoms) ----
    z = w_const + x
    q = cp.Variable(K)                       # >= (z - tau_j)_+
    r = cp.Variable(K)                       # >= 0.5 * q_j^2
    u1 = cp.Variable()                       # >= |x - a_t|

    constraints = [
        z >= tau0, z <= tauK,
        x >= 0.0, x <= (tauK - w_const),

        q >= z - taus,
        q >= 0,

        cp.square(q) <= 2 * r,   # convex ≤ affine (DPP-safe)
        r >= 0,

        u1 >=  x - a_t,  u1 >= -(x - a_t),
    ]

    # IMPORTANT: no param*param products in the objective
    # Use c1 * x (not c1 * z); the dropped c1*w_const is a parameter-only constant → irrelevant to argmin
    Fz = c1 * x + w_hinge @ r

    objective = cp.Minimize(p_t * x + eta * (u1) + Fz)
    prob = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        prob,
        parameters=[a_t, w_const, p_t, eta, w_hinge, c1],
        variables=[x],
    )

    return layer


def make_pald_flex_delivery_layer(K):
    """
    coeff = p_t * (c_delivery + eps_delivery) - p_t * c_delivery * s_prev
    """
    # ---- compile-time constants ----
    taus_full = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=float)
    taus = taus_full[:-1]            # locations of hinge knots (exclude last)
    tau0, tauK = float(taus_full[0]), float(taus_full[-1])

    # ---- decision ----
    x = cp.Variable()

    # ---- parameters (layer inputs) ----
    x_prev  = cp.Parameter(1,nonneg=True)                 # scalar
    w_const = cp.Parameter(1,nonneg=True)                 # scalar
    coeff     = cp.Parameter(1,nonneg=True)                 # delivery cost coefficienct * p_t
    delta   = cp.Parameter(1,nonneg=True)                 # scalar
    # IMPORTANT: pass these precomputed from y (outside the graph)
    w_hinge = cp.Parameter(K, nonneg=True)   # hinge weights >= 0
    c1      = cp.Parameter()                 # slope term for F (e.g., -y[0])

    # ---- auxiliaries (nonlinearities live here, no parameters inside atoms) ----
    z = w_const + x
    q = cp.Variable(K)                       # >= (z - tau_j)_+
    r = cp.Variable(K)                       # >= 0.5 * q_j^2
    u1 = cp.Variable()                       # >= |x - x_prev|
    u2 = cp.Variable()                       # >= |x|

    constraints = [
        z >= tau0, z <= tauK,
        x >= 0.0, x <= (tauK - w_const),

        q >= z - taus,
        q >= 0,

        cp.square(q) <= 2 * r,   # convex ≤ affine (DPP-safe)
        r >= 0,

        u1 >=  x - x_prev,  u1 >= -(x - x_prev),
        u2 >=  x,           u2 >= -x,
    ]

    # IMPORTANT: no param*param products in the objective
    # Use c1 * x (not c1 * z); the dropped c1*w_const is a parameter-only constant → irrelevant to argmin
    Fz = c1 * x + w_hinge @ r

    objective = cp.Minimize(coeff * x + delta * (u1 + u2) + Fz)
    prob = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        prob,
        parameters=[x_prev, w_const, coeff, delta, w_hinge, c1],
        variables=[x],
    )

    return layer


# differentiable torch objective function
# NOTE: Keep everything as torch ops; avoid Python floats that can break the graph.
def torch_objective(p_seq, a_seq, x_seq, z_seq, eta, delta, c, eps):
    """Torch version of objective_function for differentiable PALD cost.
    Inputs are torch 1D tensors of length T (float32).
    Mirrors paad_implementation.objective_function.
    """
    Tn = p_seq.shape[0]
    # state of charge s[0..T]
    s = []
    s_prev = torch.tensor(0.0, dtype=torch.float32)
    s.append(s_prev)
    for t in range(1, Tn+1):
        s_t = torch.maximum(s_prev + x_seq[t-1] - z_seq[t-1], torch.tensor(0.0))
        s.append(s_t)
        s_prev = s_t
    s_torch = torch.stack(s)  # s[0..T-1] corresponds to s_1..s_T in numpy version

    # Costs
    cost_purchasing = (p_seq * x_seq).sum()
    tracking_cost_x = eta * (x_seq - a_seq).abs().sum() if Tn > 1 else torch.tensor(0.0)
    switching_cost_z = delta * (z_seq[1:] - z_seq[:-1]).abs().sum() if Tn > 1 else torch.tensor(0.0)
    # Note: s_{t-1} term via roll(1); s_{-1} uses previous s_T but is unused because multiplied with z[0]; fix first index:
    s_prev_seq = torch.cat([s_torch[:-1]])
    discharge_cost = (p_seq * (c * z_seq + eps * z_seq - c * s_prev_seq * z_seq)).sum()
    return cost_purchasing + tracking_cost_x + switching_cost_z + discharge_cost

def hinge_from_y_torch(taus_full_t: torch.Tensor, y: torch.Tensor):
    """
    taus_full_t: torch tensor of shape (K+1,), constant breakpoints [τ0,...,τK]
    y:           torch tensor of shape (K+1,) or (B, K+1), values g(τ_j)=y_j

    Returns:
      w_hinge: (K,) or (B, K), nonneg hinge weights
      c1:      ()  or (B,),   slope term for F; c1 = -y[..., 0]
    """
    # Ensure taus on same device/dtype as y
    taus_full_t = taus_full_t.to(device=y.device, dtype=y.dtype)

    # Differences along the last dimension
    dt = taus_full_t[1:] - taus_full_t[:-1]                     # (K,)
    dy = y[..., 1:] - y[..., :-1]                               # (..., K)

    # Slopes of g on segments, then curvatures of F
    a = dy / dt                                                 # (..., K)
    s = -a                                                      # (..., K)

    # Hinge weights are curvature jumps
    w0 = s[..., :1]                                             # (..., 1)
    wj = s[..., 1:] - s[..., :-1]                               # (..., K-1)
    w = torch.cat([w0, wj], dim=-1)                             # (..., K)

    # Nonnegativity projection; subgradients pass where w>0
    w = torch.clamp(w, min=0.0)

    # c1 = -g(τ0) = -y[..., 0]
    c1 = -y[..., 0]

    return w, c1