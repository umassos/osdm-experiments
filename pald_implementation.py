import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

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
    x_prev  = cp.Parameter(1,nonneg=True)                 # scalar
    w_const = cp.Parameter(1,nonneg=True)                 # scalar
    p_t     = cp.Parameter(1,nonneg=True)                 # scalar
    gamma   = cp.Parameter(1,nonneg=True)                 # scalar
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

    objective = cp.Minimize(p_t * x + gamma * (u1 + u2) + Fz)
    prob = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        prob,
        parameters=[x_prev, w_const, p_t, gamma, w_hinge, c1],
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
    x_prev  = cp.Parameter(1,nonneg=True)                 # scalar
    w_const = cp.Parameter(1,nonneg=True)                 # scalar
    p_t     = cp.Parameter(1,nonneg=True)                 # scalar
    gamma   = cp.Parameter(1,nonneg=True)                 # scalar
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

    objective = cp.Minimize(p_t * x + gamma * (u1 + u2) + Fz)
    prob = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        prob,
        parameters=[x_prev, w_const, p_t, gamma, w_hinge, c1],
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