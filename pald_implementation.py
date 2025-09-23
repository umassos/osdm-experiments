import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

"""
PALD CVXPyLayers with integral-aware (per-segment) allocation.
"""

def make_pald_base_layer(K, gamma):
    x_parts = cp.Variable(K, nonneg=True)
    x_total = cp.Variable(nonneg=True)

    x_prev = cp.Parameter(nonneg=True)
    w_prev = cp.Parameter(nonneg=True)
    p_t = cp.Parameter()
    y_vec = cp.Parameter((K,))
    caps = cp.Parameter((K,), nonneg=True)

    constraints = [
        x_parts >= 0,
        x_parts <= caps,
        x_total == cp.sum(x_parts),
        x_total <= 1 - w_prev,
    ]
    ridge = 1e-8 * cp.sum_squares(x_parts) + 1e-8 * cp.sum_squares(x_total)
    hit_cost = p_t * x_total
    switch_cost = gamma * cp.abs(x_total - x_prev) + gamma * cp.abs(x_total)
    phi_cost = y_vec @ x_parts
    obj = cp.Minimize(hit_cost + switch_cost - phi_cost + ridge)
    prob = cp.Problem(obj, constraints)
    return CvxpyLayer(prob,
                      parameters=[x_prev, w_prev, p_t, y_vec, caps],
                      variables=[x_total])

def make_pald_flex_purchase_layer(K, gamma):
    x_parts = cp.Variable(K, nonneg=True)
    x_total = cp.Variable(nonneg=True)

    x_prev = cp.Parameter(nonneg=True)
    w_prev = cp.Parameter(nonneg=True)
    p_t = cp.Parameter()
    y_vec = cp.Parameter((K,))
    caps = cp.Parameter((K,), nonneg=True)

    constraints = [
        x_parts >= 0,
        x_parts <= caps,
        x_total == cp.sum(x_parts),
        x_total <= 1 - w_prev,
    ]
    ridge = 1e-8 * cp.sum_squares(x_parts) + 1e-8 * cp.sum_squares(x_total)
    hit_cost = p_t * x_total
    switch_cost = gamma * cp.abs(x_total - x_prev) + gamma * cp.abs(x_total)
    phi_cost = y_vec @ x_parts
    obj = cp.Minimize(hit_cost + switch_cost - phi_cost + ridge)
    prob = cp.Problem(obj, constraints)
    return CvxpyLayer(prob,
                      parameters=[x_prev, w_prev, p_t, y_vec, caps],
                      variables=[x_total])

def make_pald_flex_delivery_layer(K, delta, c_delivery, eps_delivery):
    """
    coeff = p_t * (c_delivery + eps_delivery) - p_t * c_delivery * s_prev
    """
    z_parts = cp.Variable(K, nonneg=True)
    z_total = cp.Variable(nonneg=True)

    z_prev = cp.Parameter(nonneg=True)
    v_prev = cp.Parameter(nonneg=True)
    coeff = cp.Parameter()
    y_vec = cp.Parameter((K,))
    caps = cp.Parameter((K,), nonneg=True)

    constraints = [
        z_parts >= 0,
        z_parts <= caps,
        z_total == cp.sum(z_parts),
        z_total <= 1 - v_prev,
    ]
    ridge = 1e-8 * cp.sum_squares(z_parts) + 1e-8 * cp.sum_squares(z_total)
    hit_cost = coeff * z_total
    switch_cost = delta * cp.abs(z_total - z_prev) + delta * cp.abs(z_total)
    phi_cost = y_vec @ z_parts
    obj = cp.Minimize(hit_cost + switch_cost - phi_cost + ridge)
    prob = cp.Problem(obj, constraints)
    return CvxpyLayer(prob,
                      parameters=[z_prev, v_prev, coeff, y_vec, caps],
                      variables=[z_total])