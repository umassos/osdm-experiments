
import torch
import cvxpy as cp

def project_y_robust(y_tensor: torch.Tensor, K: int, p_min: float, p_max: float,
                     gamma: float, delta: float, c: float, eps: float, T: int,
                     beta: float = 6.0) -> list:
    """Project y onto robustness set R_b^(beta) with discretization w_j = j/K.
    Solve QP: minimize ||y_var - y||^2 subject to
      - p_min <= y_i <= p_max
      - y is non-increasing
      - y[K-1] <= p_min + 2*gamma
      - For each j in {0..K}: ∫_0^{w_j} psi + (1-w_j)(p_max+2γ) + p_max(c+eps) + 2δ - c w_j p_min
          <= beta * (psi(w_j) - 2γ + eps p_max + (2δ+2γ)/T)
      where psi is piecewise-constant over K equal segments, and the integral
      is approximated by left Riemann sum using segment widths 1/K.
    Returns a Python list of projected y values.
    """
    y_np = y_tensor.detach().cpu().numpy().astype(float)
    y_var = cp.Variable(K)
    constraints = []
    # Bounds
    for i in range(K):
        constraints += [y_var[i] >= p_min, y_var[i] <= p_max]
    # Monotonic non-increasing
    for i in range(1, K):
        constraints += [y_var[i] <= y_var[i-1]]
    # Last value constraint
    constraints += [y_var[K-1] <= float(p_min) + 2.0 * float(gamma)]
    # Discretized robustness inequalities at w_j = j/K
    for j in range(0, K+1):
        w = j / K
        # Integral approx: (1/K) * sum_{i=0}^{j-1} y_i
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
        # psi(w_j): take left value y_{j-1} for j>=1; for j=0, use y_0
        psi_w = y_var[j-1] if j >= 1 else y_var[0]
        rhs = beta * (psi_w - 2 * gamma + eps * p_max + (2 * delta + 2 * gamma) / T)
        constraints += [lhs <= rhs]
    # Objective: squared distance to current y
    obj = cp.Minimize(cp.sum_squares(y_var - y_np))
    prob = cp.Problem(obj, constraints)
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, verbose=False, warm_start=True)
        if y_var.value is None:
            return y_np.tolist()
        return y_var.value.tolist()
    except Exception:
        return y_np.tolist()

def project_y_flex_robust(y_phi_tensor: torch.Tensor,
                          y_psi_tensor: torch.Tensor,
                          K: int, p_min: float, p_max: float,
                          gamma: float, delta: float,
                          c: float, eps: float, T: int,
                          beta: float = 6.0) -> tuple[list, list]:
    """
    Project (phi, psi) onto the flexible robustness set R_f^(beta) using a trapezoid-rule discretization.
      Variables: phi, psi ∈ R^K (segment values, treated as piecewise-constant per segment).
      Grid: w_j = j/K, v_l = l/K for j,l ∈ {0..K}.
      Nodes (for trapezoid): phi_nodes[j] and psi_nodes[l] are affine in (phi, psi).
      Integrals: cumulative trapezoid sums up to w_j, v_l.

    Constraints:
      - p_min ≤ phi[i] ≤ p_max; p_min ≤ psi[i] ≤ p_max
      - phi, psi non-increasing
      - phi[-1] ≤ p_min + 2γ
      - psi[-1] ≤ p_min (c+ε) + 2δ
      - For all j,l: I_phi(w_j) + (1-w_j)(p_max+2γ) - c w_j p_min
                     + I_psi(v_l) + (1-v_l)(p_max(c+ε)+2δ)
                     ≤ β [ ((1+ε)/(1+c+ε)) (phi(w_j) + psi(v_l) - 2κ) + 2κ/T ],
        where κ = γ + δ, and phi(w_j), psi(v_l) are node values.

    Returns (phi_proj, psi_proj) as Python lists.
    """
    y_phi = y_phi_tensor.detach().cpu().numpy().astype(float)
    y_psi = y_psi_tensor.detach().cpu().numpy().astype(float)

    phi = cp.Variable(K)
    psi = cp.Variable(K)

    cons = []
    # bounds
    for i in range(K):
        cons += [phi[i] >= p_min, phi[i] <= p_max]
        cons += [psi[i] >= p_min, psi[i] <= p_max]
    # monotone non-increasing
    for i in range(1, K):
        cons += [phi[i] <= phi[i-1]]
        cons += [psi[i] <= psi[i-1]]
    # endpoint constraints
    cons += [phi[K-1] <= p_min + 2.0 * gamma]
    cons += [psi[K-1] <= p_min * (c + eps) + 2.0 * delta]

    # trapezoid nodes (affine in variables)
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

    # cumulative trapezoid integrals up to each node
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

    # robustness inequalities over the 2D grid
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

    # projection objective
    obj = cp.Minimize(cp.sum_squares(phi - y_phi) + cp.sum_squares(psi - y_psi))
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, verbose=False, warm_start=True)
        if phi.value is None or psi.value is None:
            return y_phi.tolist(), y_psi.tolist()
        return phi.value.tolist(), psi.value.tolist()
    except Exception:
        return y_phi.tolist(), y_psi.tolist()