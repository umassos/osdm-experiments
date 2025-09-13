import numpy as np



# implements the OSDM objective function outside of Gurobi 
def objective_function(
    T: int,
    p: list[float],
    gamma: float,
    delta: float,
    c_delivery: float,
    eps_delivery: float,
    x: list[float],
    z: list[float]
) -> float:
    """
    Computes the objective function value for the OSDM problem.

    Args:
        T (int): The time horizon.
        p (list): A list of prices p_t for t in [1, T].
        gamma (float): Switching cost parameter for x.
        delta (float): Switching cost parameter for z.
        c_delivery (float): Parameter for the discharge cost function.
        eps_delivery (float): Parameter for the discharge cost function (eps_delivery < c_delivery).
        x (list): A list of charge values x_t for t in [1, T].
        z (list): A list of discharge values z_t for t in [1, T].

    Returns:
        float: The computed objective function value.
    """
    if not (len(p) == T and len(x) == T and len(z) == T and len(s) == T + 1):
        raise ValueError("Length of price, x, and z lists must equal T.")

    obj_value = 0.0

    # state of charge variable (starts at 0)
    s = np.zeros(T + 1)
    for t in range(1, T + 1):
        s[t] = max(s[t - 1] + x[t - 1] - z[t - 1], 0)

    # Cost of purchasing energy
    cost_purchasing = sum(p[t] * z[t] for t in range(T))
    obj_value += cost_purchasing

    # Switching costs (L1 norm)
    switching_cost_x = sum(gamma * abs(x[t] - x[t - 1]) for t in range(1, T))
    switching_cost_z = sum(delta * abs(z[t] - z[t - 1]) for t in range(1, T))
    obj_value += (switching_cost_x + switching_cost_z)

    # Discharge costs
    discharge_cost = sum(c_delivery * z[t] + eps_delivery * z[t] - c_delivery * z[t] * s[t-1] for t in range(T))
    obj_value += discharge_cost

    return obj_value