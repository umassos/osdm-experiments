import gurobipy as gp
from gurobipy import GRB


def optimal_solution(
    T: int,
    p: list[float],
    gamma: float,
    delta: float,
    c_delivery: float,
    eps_delivery: float,
    S: float,
    b: list[float],
    f: list[float],
    Delta_f: list[float],
    s_0: float = 0.0,
    x_0: float = 0.0,
    z_0: float = 0.0,
    solver_timeout: float = 120.0
):
    """
    Solves an offline version of the OSDM problem using Gurobi.

    Args:
        T (int): The time horizon.
        p (list): A list of prices p_t for t in [1, T].
        gamma (float): Ramping cost parameter for x.
        delta (float): Ramping cost parameter for z.
        c_delivery (float): Parameter for the discharge cost function.
        eps_delivery (float): Parameter for the discharge cost function (eps_delivery < c_delivery).
        S (float): Maximum capacity of the inventory.
        b (list): A list of lower bounds b_t for z_t for t in [1, T]. (base demand values)
        f (list): A list of values f_t for the upper bound on z_t. (flexible demand values)
        Delta_f (list): A list of deadline values -- f[i] must be delivered by time Delta_f[i].
        s_0 (float): Initial state of the inventory.
        x_0 (float): Initial charge value at t=0.
        z_0 (float): Initial discharge value at t=0.

    Returns:
        A tuple containing:
        - status (str): The optimization status of the model.
        - results (dict): A dictionary with the optimal values for x, z, s,
                          and the objective value. Returns None if no solution is found.
    """
    if not (len(p) == T and len(b) == T and len(f) == T and len(Delta_f) == T):
        raise ValueError("Length of price and bound lists must equal T.")

    # --- Model Setup ---
    model = gp.Model("InventoryOptimization")

    # Time indices from 1 to T for easier mapping from mathematical model
    time_steps = range(1, T + 1)
    # State indices from 0 to T to include initial state
    state_steps = range(0, T + 1)
    
    # Map list inputs to Gurobi-friendly dictionaries indexed by time t
    p_map = {t: p[t-1] for t in time_steps}
    b_map = {t: b[t-1] for t in time_steps}
    f_map = {t: f[t-1] for t in time_steps}
    Delta_f_map = {t: Delta_f[t-1] for t in time_steps}
    f_cumulative = {t: sum(f_map[tau] for tau in range(1, t + 1)) for t in time_steps}


    # --- Variable Declaration ---
    x = model.addVars(time_steps, name="x", lb=0.0)
    z = model.addVars(time_steps, name="z", lb=0.0)
    s = model.addVars(state_steps, name="s", lb=0.0, ub=S)
    dx_abs = model.addVars(time_steps, name="dx_abs", lb=0.0)
    dz_abs = model.addVars(time_steps, name="dz_abs", lb=0.0)

    # --- Constraints ---
    
    # Initial state constraint
    model.addConstr(s[0] == s_0, name="initial_state")

    # State transition equation: s_t = s_{t-1} + x_t - z_t
    model.addConstrs(
        (s[t] == s[t-1] + x[t] - z[t] for t in time_steps), name="state_update"
    )

    # Bounds on z_t: b_t <= z_t <= b_t + sum_{tau=1 to t} f_tau
    model.addConstrs(
        (z[t] >= b_map[t] for t in time_steps), name="z_lower_bound"
    )
    model.addConstrs(
        (z[t] <= b_map[t] + f_cumulative[t] for t in time_steps), name="z_upper_bound"
    )

    # Bounds on x_t, the cumulative x_t up to time t is allowed to be at most S + cumulative f_t + cumulative b_t up to t
    model.addConstrs(
        (gp.quicksum(x[tau] for tau in range(1, t + 1)) <= S + gp.quicksum(f_map[tau] for tau in range(1, t + 1)) + gp.quicksum(b_map[tau] for tau in range(1, t + 1))
         for t in time_steps),
        name="x_cumulative_upper_bound"
    )

    # one additional lower bound on z_t -- the cumulative sum of z_t up to time t should be at least
    # the cumulative sum of all b_t (satisfied by z_lower_bound constraint) plus all f_t where Delta_f_t <= t
    model.addConstrs(
        (gp.quicksum(z[tau] for tau in range(1, t + 1)) >=
         gp.quicksum(b_map[tau] for tau in range(1, t + 1)) +
         gp.quicksum(f_map[tau] for tau in time_steps if Delta_f_map[tau] <= t)
         for t in time_steps),
        name="z_cumulative_lower_bound"
    )
    
    # Absolute value linearization constraints
    model.addConstr(dx_abs[1] >= x[1] - x_0, name="dx_abs_pos_t1")
    model.addConstr(dx_abs[1] >= -(x[1] - x_0), name="dx_abs_neg_t1")
    model.addConstr(dz_abs[1] >= z[1] - z_0, name="dz_abs_pos_t1")
    model.addConstr(dz_abs[1] >= -(z[1] - z_0), name="dz_abs_neg_t1")
    model.addConstrs(
        (dx_abs[t] >= x[t] - x[t-1] for t in time_steps if t > 1), name="dx_abs_pos"
    )
    model.addConstrs(
        (dx_abs[t] >= -(x[t] - x[t-1]) for t in time_steps if t > 1), name="dx_abs_neg"
    )
    model.addConstrs(
        (dz_abs[t] >= z[t] - z[t-1] for t in time_steps if t > 1), name="dz_abs_pos"
    )
    model.addConstrs(
        (dz_abs[t] >= -(z[t] - z[t-1]) for t in time_steps if t > 1), name="dz_abs_neg"
    )


    # --- Objective Function ---
    linear_obj = gp.quicksum(
        p_map[t] * x[t] +
        gamma * dx_abs[t] +
        delta * dz_abs[t] +
        c_delivery * p_map[t] * z[t] +
        eps_delivery * p_map[t] * z[t]
        for t in time_steps
    )
    
    quadratic_obj = gp.quicksum(
        -1 * c_delivery * p_map[t] * s[t-1] * z[t]
        for t in time_steps
    )

    model.setObjective(linear_obj + quadratic_obj, GRB.MINIMIZE)

    # --- Solver Configuration ---
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    # set a time limit of 60 seconds
    model.setParam('TimeLimit', solver_timeout)

    # --- Solve and Post-process ---
    model.optimize()

    status = GRB.Status.OPTIMAL
    if model.Status == status:
        results = {
            "x": [x[t].X for t in time_steps],
            "z": [z[t].X for t in time_steps],
            "s": [s[t].X for t in state_steps],
            "obj_val": model.ObjVal
        }
        return "Optimal", results
    else:
        return model.Status, None
    


def optimal_tracking_solution(
    T: int,
    p: list[float],
    eta: float,
    delta: float,
    c_delivery: float,
    eps_delivery: float,
    S: float,
    b: list[float],
    f: list[float],
    Delta_f: list[float],
    a: list[float],
    s_0: float = 0.0,
    x_0: float = 0.0,
    z_0: float = 0.0,
    solver_timeout: float = 120.0
):
    """
    Solves an offline version of the OSDM problem using Gurobi.

    Args:
        T (int): The time horizon.
        p (list): A list of prices p_t for t in [1, T].
        eta (float): Tracking cost parameter for x.
        delta (float): Ramping cost parameter for z.
        c_delivery (float): Parameter for the discharge cost function.
        eps_delivery (float): Parameter for the discharge cost function (eps_delivery < c_delivery).
        S (float): Maximum capacity of the inventory.
        b (list): A list of lower bounds b_t for z_t for t in [1, T]. (base demand values)
        f (list): A list of values f_t for the upper bound on z_t. (flexible demand values)
        Delta_f (list): A list of deadline values -- f[i] must be delivered by time Delta_f[i].
        a (list): A list of tracking target values a_t for t in [1, T].
        s_0 (float): Initial state of the inventory.
        x_0 (float): Initial charge value at t=0.
        z_0 (float): Initial discharge value at t=0.

    Returns:
        A tuple containing:
        - status (str): The optimization status of the model.
        - results (dict): A dictionary with the optimal values for x, z, s,
                          and the objective value. Returns None if no solution is found.
    """
    if not (len(p) == T and len(b) == T and len(f) == T and len(Delta_f) == T):
        raise ValueError("Length of price and bound lists must equal T.")

    # --- Model Setup ---
    model = gp.Model("InventoryOptimization")

    # Time indices from 1 to T for easier mapping from mathematical model
    time_steps = range(1, T + 1)
    # State indices from 0 to T to include initial state
    state_steps = range(0, T + 1)
    
    # Map list inputs to Gurobi-friendly dictionaries indexed by time t
    p_map = {t: p[t-1] for t in time_steps}
    b_map = {t: b[t-1] for t in time_steps}
    f_map = {t: f[t-1] for t in time_steps}
    Delta_f_map = {t: Delta_f[t-1] for t in time_steps}
    f_cumulative = {t: sum(f_map[tau] for tau in range(1, t + 1)) for t in time_steps}


    # --- Variable Declaration ---
    x = model.addVars(time_steps, name="x", lb=0.0)
    z = model.addVars(time_steps, name="z", lb=0.0)
    s = model.addVars(state_steps, name="s", lb=0.0, ub=S)
    dx_abs = model.addVars(time_steps, name="dx_abs", lb=0.0)
    dz_abs = model.addVars(time_steps, name="dz_abs", lb=0.0)

    # --- Constraints ---
    
    # Initial state constraint
    model.addConstr(s[0] == s_0, name="initial_state")

    # State transition equation: s_t = s_{t-1} + x_t - z_t
    model.addConstrs(
        (s[t] == s[t-1] + x[t] - z[t] for t in time_steps), name="state_update"
    )

    # Bounds on z_t: b_t <= z_t <= b_t + sum_{tau=1 to t} f_tau
    model.addConstrs(
        (z[t] >= b_map[t] for t in time_steps), name="z_lower_bound"
    )
    model.addConstrs(
        (z[t] <= b_map[t] + f_cumulative[t] for t in time_steps), name="z_upper_bound"
    )

    # Bounds on x_t, the cumulative x_t up to time t is allowed to be at most S + cumulative f_t + cumulative b_t up to t
    model.addConstrs(
        (gp.quicksum(x[tau] for tau in range(1, t + 1)) <= S + gp.quicksum(f_map[tau] for tau in range(1, t + 1)) + gp.quicksum(b_map[tau] for tau in range(1, t + 1))
         for t in time_steps),
        name="x_cumulative_upper_bound"
    )

    # one additional lower bound on z_t -- the cumulative sum of z_t up to time t should be at least
    # the cumulative sum of all b_t (satisfied by z_lower_bound constraint) plus all f_t where Delta_f_t <= t
    model.addConstrs(
        (gp.quicksum(z[tau] for tau in range(1, t + 1)) >=
         gp.quicksum(b_map[tau] for tau in range(1, t + 1)) +
         gp.quicksum(f_map[tau] for tau in time_steps if Delta_f_map[tau] <= t)
         for t in time_steps),
        name="z_cumulative_lower_bound"
    )
    
    # Absolute value linearization constraints
    model.addConstr(dz_abs[1] >= z[1] - z_0, name="dz_abs_pos_t1")
    model.addConstr(dz_abs[1] >= -(z[1] - z_0), name="dz_abs_neg_t1")
    model.addConstrs(
        (dz_abs[t] >= z[t] - z[t-1] for t in time_steps if t > 1), name="dz_abs_pos"
    )
    model.addConstrs(
        (dz_abs[t] >= -(z[t] - z[t-1]) for t in time_steps if t > 1), name="dz_abs_neg"
    )
    model.addConstrs(
        (dx_abs[t] >= x[t] - a[t-1] for t in time_steps), name="dx_abs_pos"
    )
    model.addConstrs(
        (dx_abs[t] >= -(x[t] - a[t-1]) for t in time_steps), name="dx_abs_neg"
    )


    # --- Objective Function ---
    linear_obj = gp.quicksum(
        p_map[t] * x[t] +
        eta * dx_abs[t] +
        delta * dz_abs[t] +
        c_delivery * p_map[t] * z[t] +
        eps_delivery * p_map[t] * z[t]
        for t in time_steps
    )
    
    quadratic_obj = gp.quicksum(
        -1 * c_delivery * p_map[t] * s[t-1] * z[t]
        for t in time_steps
    )

    model.setObjective(linear_obj + quadratic_obj, GRB.MINIMIZE)

    # --- Solver Configuration ---
    model.setParam('OutputFlag', 0)
    model.setParam('NonConvex', 2)
    # set a time limit of 60 seconds
    model.setParam('TimeLimit', solver_timeout)

    # --- Solve and Post-process ---
    model.optimize()

    status = GRB.Status.OPTIMAL
    if model.Status == status:
        results = {
            "x": [x[t].X for t in time_steps],
            "z": [z[t].X for t in time_steps],
            "s": [s[t].X for t in state_steps],
            "obj_val": model.ObjVal
        }
        return "Optimal", results
    else:
        return model.Status, None