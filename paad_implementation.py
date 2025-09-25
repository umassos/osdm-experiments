import numpy as np
import cvxpy as cp
from scipy.special import lambertw
import scipy.integrate as integrate

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
    if not (len(p) == T and len(x) == T and len(z) == T):
        raise ValueError("Length of price, x, and z lists must equal T.")

    obj_value = 0.0

    # state of charge variable (starts at 0)
    s = np.zeros(T + 1)

    # purchasing variable (starts at 0)
    x_new = np.zeros(T + 1)
    x_new[1:] = np.array(x)

    # delivery variable (starts at 0)
    z_new = np.zeros(T + 1)
    z_new[1:] = np.array(z)

    for t in range(1, T+1):       
        s[t] = max(s[t - 1] + x_new[t] - z_new[t], 0)

    # Cost of purchasing energy
    cost_purchasing = sum(p[t-1] * x_new[t] for t in range(1, T+1))
    obj_value += cost_purchasing

    # Switching costs (L1 norm)
    switching_cost_x = sum(gamma * abs(x_new[t] - x_new[t - 1]) for t in range(1, T+1))
    switching_cost_z = sum(delta * abs(z_new[t] - z_new[t - 1]) for t in range(1, T+1))
    obj_value += (switching_cost_x + switching_cost_z)

    # Discharge costs.   # p_1 * (c_delivery * z_1 + eps_delivery * z_1 - c_delivery * z_1 * s_0) + ...   
    discharge_cost = sum(p[t-1] * (c_delivery * z_new[t] + eps_delivery * z_new[t] - c_delivery * z_new[t] * s[t-1]) for t in range(1, T+1))
    obj_value += discharge_cost

    return obj_value


# define object for each ''driver''
class BaseDriver:
    def __init__(self, id: int, b: float, gamma: float, delta: float, alpha: float, p_min: float, p_max: float, c_delivery: float, eps_delivery: float, T: int):
        self.id = id
        self.b = b
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.T = T
        self.p_min = p_min
        self.p_max = p_max
        self.c_delivery = c_delivery
        self.eps_delivery = eps_delivery

        # pseudo-decision accounting
        self.prev_decision = 0.0
        self.pseudo_decision = 0.0

        # progress accounting
        self.w = 0.0 # cumulative amount purchased so far
    
    def threshold(self, w: float):
        p_min, p_max, alpha, gamma, delta, T = self.p_min, self.p_max, self.alpha, self.gamma, self.delta, self.T
        c, eps = self.c_delivery, self.eps_delivery
        b = self.b
        lhs = p_max + 2*gamma + p_min *c
        inside_exp = (p_max * (1 + c+ eps) + 2*(gamma+delta))/(alpha) - p_max*(1+eps) - p_min * c - 2*(gamma+delta)/T
        rhs = (inside_exp)*np.exp((w)/(alpha*b))
        return lhs + rhs

    def thresholdAntiDeriv(self, w: float):
        p_min, p_max, alpha, gamma, delta, T = self.p_min, self.p_max, self.alpha, self.gamma, self.delta, self.T
        c, eps = self.c_delivery, self.eps_delivery
        b = self.b
        lhs = (p_max + 2*gamma + p_min *c)*w
        inside_exp = (p_max * (1 + c+ eps) + 2*(gamma+delta))/(alpha) - (p_max*(1+eps) + p_min * c + 2*(gamma+delta)/T)
        rhs = (inside_exp)*alpha*b*cp.exp((w)/(alpha*b))
        return lhs + rhs

    def set_pseudo_decision(self, excess: float, total: float):
        ratio = self.prev_decision / total if total > 0 else 0
        self.pseudo_decision = self.prev_decision + max(0.0, excess * ratio)

    def get_x_decision(self, curr_price: float):
        x = cp.Variable(nonneg=True)
        constraints = [0 <= x, x <= (self.b - self.w)]
        prob = cp.Problem(cp.Minimize(self.pcBaseMinimization(x, curr_price)), constraints)
        prob.solve(solver=cp.CLARABEL)
        target = x.value

        # check if target is none
        if target is None:
            target = 0.0
        
        self.prev_decision = target
        self.w += target
        return target
    
    def pcBaseMinimization(self, x, curr_price):
        prev = self.pseudo_decision
        hit_cost = (curr_price * x)
        switch_cost = self.gamma * cp.abs(x - prev) + self.gamma * cp.abs(x)
        # pseudo_cost = integrate.quad(self.threshold, self.w, self.w + x)[0]
        pseudo_cost = self.thresholdAntiDeriv(self.w + x) - self.thresholdAntiDeriv(self.w)
        return hit_cost + switch_cost - pseudo_cost



class FlexibleDriver:
    def __init__(self, id: int, f: float, delta_f: float, gamma: float, delta: float, alpha: float, p_min: float, p_max: float, c_delivery: float, eps_delivery: float, T: int):
        self.id = id
        self.f = f
        self.delta_f = delta_f
        self.gamma = gamma
        self.delta = delta
        self.alpha = alpha
        self.T = T
        self.p_min = p_min
        self.p_max = p_max
        self.c_delivery = c_delivery
        self.eps_delivery = eps_delivery

        # compute alpha prime
        self.alpha_prime = self.alpha / ((1+self.c_delivery+self.eps_delivery)/(1+self.eps_delivery))

        # pseudo-decision accounting
        self.prev_purchasing_decision = 0.0
        self.prev_delivery_decision = 0.0
        self.pseudo_purchasing_decision = 0.0
        self.pseudo_delivery_decision = 0.0

        # progress accounting
        self.w = 0.0 # cumulative amount purchased so far
        self.w_last = 0.0 # last cumulative amount purchased (for threshold calculation)
        self.v = 0.0 # cumulative amount delivered so far

    def set_purchasing_pseudo_decision(self, excess: float, total: float):
        ratio = self.prev_purchasing_decision / total if total > 0 else 0
        self.pseudo_purchasing_decision = self.prev_purchasing_decision + max(0.0, excess*ratio)

    def set_delivery_pseudo_decision(self, excess: float, total: float):
        ratio = self.prev_delivery_decision / total if total > 0 else 0
        self.pseudo_delivery_decision = self.prev_delivery_decision + max(0.0, excess*ratio)

    def purchasing_threshold(self, w: float):
        p_min, p_max, alpha, gamma, T = self.p_min, self.p_max, self.alpha_prime, self.gamma, self.T
        c, eps = self.c_delivery, self.eps_delivery
        omega = (1 + c + eps)/(1 + eps)
        f = self.f
        lhs = p_max + 2*gamma + p_min *c
        inside_exp = (p_max + 2*(gamma))/(alpha) - (p_max + p_min * c + ((2*(gamma)/T) * omega))
        rhs = (inside_exp)*np.exp((w)/(alpha*f))
        return lhs + rhs
    
    def purchasing_thresholdAntiDeriv(self, w: float):
        p_min, p_max, alpha, gamma, T = self.p_min, self.p_max, self.alpha_prime, self.gamma, self.T
        c, eps = self.c_delivery, self.eps_delivery
        omega = (1 + c + eps)/(1 + eps)
        f = self.f
        lhs = (p_max + 2*gamma + p_min *c)*w
        inside_exp = (p_max + 2*(gamma))/(alpha) - (p_max + p_min * c + ((2*(gamma)/T) * omega))
        rhs = (inside_exp)*alpha*f*cp.exp((w)/(alpha*f))
        return lhs + rhs

    def set_pseudo_decision(self, excess: float):
        self.pseudo_decision = self.prev_decision + max(0.0, excess)
    
    def get_x_decision(self, curr_price: float, time_step: int):
        x = cp.Variable(nonneg=True)
        constraints = [0 <= x, x <= (self.f - self.w)]
        prob = cp.Problem(cp.Minimize(self.pcPurchaseMinimization(x, curr_price)), constraints)
        prob.solve(solver=cp.CLARABEL)
        target = x.value

        # check if target is none
        if target is None:
            target = 0.0

        self.prev_purchasing_decision = target
        self.w_last = self.w
        self.w += target

        # Force full purchase by (and at) the deadline (align with delivery convention: >= delta_f - 1)
        if time_step >= self.delta_f - 1:
            extra = self.f - self.w
            if extra > 0:
                self.prev_purchasing_decision += extra
                self.w += extra
            # ensure exactly w = f
            self.w = self.f

        return self.prev_purchasing_decision
    
    def pcPurchaseMinimization(self, x, curr_price):
        prev = self.pseudo_purchasing_decision
        hit_cost = (curr_price * x)
        switch_cost = self.gamma * cp.abs(x - prev) + self.gamma * cp.abs(x)
        # pseudo_cost = integrate.quad(self.purchasing_threshold, self.w, self.w + x)[0]
        pseudo_cost = self.purchasing_thresholdAntiDeriv(self.w + x) - self.purchasing_thresholdAntiDeriv(self.w)
        return hit_cost + switch_cost - pseudo_cost
    
    def delivery_threshold(self, v: float):
        p_max, alpha, delta, T = self.p_max, self.alpha_prime, self.delta, self.T
        c, eps = self.c_delivery, self.eps_delivery
        omega = (1 + c + eps)/(1 + eps)
        f = self.f
        lhs = p_max*(c+eps) + 2*delta
        inside_exp = (p_max*(c+eps) + 2*(delta))/(alpha) - (p_max*(c+eps) + ((2*(delta)/T) * omega))
        rhs = (inside_exp)*np.exp((v)/(alpha*f))
        return lhs + rhs
    
    def delivery_thresholdAntiDeriv(self, v: float):
        p_max, alpha, delta, T = self.p_max, self.alpha_prime, self.delta, self.T
        c, eps = self.c_delivery, self.eps_delivery
        omega = (1 + c + eps)/(1 + eps)
        f = self.f
        lhs = (p_max*(c+eps) + 2*delta)*v
        inside_exp = (p_max*(c+eps) + 2*(delta))/(alpha) - (p_max*(c+eps) + ((2*delta)/T) * omega)
        rhs = (inside_exp)*alpha*f*cp.exp((v)/(alpha*f))
        return lhs + rhs

    def get_z_decision(self, curr_price: float, time_step: int, storage_state: float):
        z = cp.Variable(nonneg=True)
        constraints = [0 <= z, z <= (self.f - self.v)]

        # If deadline reached/passed, deliver all remaining, regardless of prior purchases.
        if time_step >= self.delta_f - 1:
            target = self.f - self.v
            # record and update
            self.prev_delivery_decision = target
            self.v = self.f
            return target

        # Price-triggered shortcut (only before deadline): deliver purchased remainder if beneficial
        # Note: compares against current delivery threshold
        if self.w > self.v and curr_price * (self.c_delivery + self.eps_delivery - self.c_delivery * storage_state) <= self.delivery_threshold(self.v):
            target = self.w - self.v
            self.prev_delivery_decision = target
            self.v += target
            return target

        # Otherwise, solve CVX and cap by purchased remainder
        prob = cp.Problem(cp.Minimize(self.pcDeliveryMinimization(z, curr_price, storage_state)), constraints)
        prob.solve(solver=cp.CLARABEL)
        target = z.value if z.value is not None else 0.0

        # cannot deliver more than purchased so far (before deadline)
        target = min(target, max(0.0, self.w - self.v))

        self.prev_delivery_decision = target
        self.v += target
        return target

    def pcDeliveryMinimization(self, z, curr_price, storage_state):
        prev = self.pseudo_delivery_decision
        c, eps = self.c_delivery, self.eps_delivery
        hit_cost = (curr_price * (c + eps) - curr_price * c * storage_state) * z
        switch_cost = self.delta * cp.abs(z - prev) + self.delta * cp.abs(z)
        pseudo_cost = self.delivery_thresholdAntiDeriv(self.v + z) - self.delivery_thresholdAntiDeriv(self.v)
        return hit_cost + switch_cost - pseudo_cost

def get_alpha(p_min: float, p_max: float, c: float, eps: float, T: int, gamma: float, delta: float) -> float:
    """
    Computes the competitive ratio alpha for the PAAD algorithm.
    """
    omega = (1 + c + eps) / (1 + eps)
    inner_frac = ((omega*(2*(gamma+delta)/T) - p_min*c - (1+c+eps)*p_max)/((1+c+eps)*p_max + 2*(gamma+delta)))
    numerator = ((1+c+eps)*p_max - (1+eps)*p_min) * np.exp(inner_frac)
    denominator = -1* ((1+c+eps)*p_max + 2*(gamma+delta))
    W_term = lambertw(numerator/denominator, k=0).real
    alpha = (omega) / (W_term - inner_frac)

    return alpha

def paad_algorithm(
    T: int,
    p: list[float],
    gamma: float,
    delta: float,
    c_delivery: float,
    eps_delivery: float,
    p_min: float,
    p_max: float,
    S: float,
    b: list[float],
    f: list[float],
    Delta_f: list[float],
    s_0: float = 0.0,
    x_0: float = 0.0,
    z_0: float = 0.0
):
    """
    Generates the "online" solution of the PAAD algorithm using CVXPY.

    Args:
        T (int): The time horizon.
        p (list): A list of prices p_t for t in [1, T].
        gamma (float): Ramping cost parameter for x.
        delta (float): Ramping cost parameter for z.
        c_delivery (float): Parameter for the discharge cost function.
        eps_delivery (float): Parameter for the discharge cost function (eps_delivery < c_delivery).
        p_min (float): Minimum price for the signal.
        p_max (float): Maximum price for the signal.
        S (float): Maximum capacity of the inventory.
        b (list): A list of lower bounds b_t for z_t for t in [1, T]. (base demand values)
        f (list): A list of values f_t for the upper bound on z_t. (flexible demand values)
        Delta_f (list): A list of deadline values -- f[i] must be delivered by time Delta_f[i].
        s_0 (float): Initial state of the inventory.
        x_0 (float): Initial charge value at t=0.
        z_0 (float): Initial discharge value at t=0.

    Returns:
        - results (dict): A dictionary with the PAAD values for x, z, s,
                          and the objective value. Returns None if no solution is found.
    """

    if not (len(p) == T and len(b) == T and len(f) == T and len(Delta_f) == T):
        raise ValueError("Length of price and bound lists must equal T.")
    
    x_sol = [x_0]
    z_sol = [z_0]
    s_sol = [s_0]
    base_drivers = set()
    flexible_drivers = set()

    # get value of alpha (competitive ratio)
    alpha = get_alpha(p_min, p_max, c_delivery, eps_delivery, 96, gamma, delta) 
    # print(f"Computed alpha: {alpha}")

    # add base driver with size S to the system at initialization
    new_driver = BaseDriver(id=0, b=S, gamma=gamma, delta=delta, alpha=alpha, p_min=p_min, p_max=p_max, c_delivery=c_delivery, eps_delivery=eps_delivery, T=T)
    base_drivers.add(new_driver)

    # simulate behavior of online algorithm using a for loop
    for (t, curr_price) in enumerate(p):
        b_t = b[t]
        f_t = f[t]
        delta_f_t = Delta_f[t]

        # if base demand is non-zero, add a new base driver
        if b_t > 0:
            new_driver = BaseDriver(id=(t+1)*2, b=b_t, gamma=gamma, delta=delta, alpha=alpha, p_min=p_min, p_max=p_max, c_delivery=c_delivery, eps_delivery=eps_delivery, T=T)
            base_drivers.add(new_driver)
        # if flexible demand is non-zero, add a new flexible driver
        if f_t > 0:
            new_driver = FlexibleDriver(id=(t+1)*2+1, f=f_t, delta_f=delta_f_t, gamma=gamma, delta=delta, alpha=alpha, p_min=p_min, p_max=p_max, c_delivery=c_delivery, eps_delivery=eps_delivery, T=T)
            flexible_drivers.add(new_driver)
        # compute pseudo-decisions
        prev_purchasing_decisions = 0.0
        prev_delivery_decisions = 0.0

        for driver in base_drivers:
            prev_purchasing_decisions += driver.prev_decision
        for driver in flexible_drivers:
            prev_purchasing_decisions += driver.prev_purchasing_decision
            prev_delivery_decisions += driver.prev_delivery_decision
        purchasing_excess = x_sol[-1] - prev_purchasing_decisions
        delivery_excess = z_sol[-1] - prev_delivery_decisions
        for driver in base_drivers:
            driver.set_pseudo_decision(purchasing_excess, x_sol[-1])
        for driver in flexible_drivers:
            driver.set_purchasing_pseudo_decision(purchasing_excess, x_sol[-1])
            driver.set_delivery_pseudo_decision(delivery_excess, z_sol[-1])

        # compute new decisions
        base_x_decisions = []
        flexible_x_decisions = []
        z_decisions = [b_t]  # start with base demand

        # for each base driver, compute its x decision
        for driver in base_drivers:
            x = driver.get_x_decision(curr_price)
            base_x_decisions.append(x)
        
        # for each flexible driver, compute its x and z decisions
        for driver in flexible_drivers:
            x = driver.get_x_decision(curr_price, t)
            flexible_x_decisions.append(x)
            storage_state = s_sol[-1]/S if S > 0 else 0.0
            z = driver.get_z_decision(curr_price, t, storage_state)
            z_decisions.append(z)

        # aggregate global decisions
        x_t = sum(base_x_decisions) + sum(flexible_x_decisions)
        z_t = sum(z_decisions)

        # check that enough has been purchased to cover deliveries
        x_t = max(x_t, z_t - s_sol[-1])

        # update the storage state of the system
        s_t = s_sol[-1] + x_t - z_t
        s_sol.append(s_t)

        # save decisions
        x_sol.append(x_t)
        z_sol.append(z_t)

        # if the storage is empty, remove all base drivers and initialize a new one with size S
        if s_t <= 0.0:
            base_drivers = set()
            new_driver = BaseDriver(id=0, b=S, gamma=gamma, delta=delta, alpha=alpha, p_min=p_min, p_max=p_max, c_delivery=c_delivery, eps_delivery=eps_delivery, T=T)
            base_drivers.add(new_driver)

    # Store the total evolution of the system
    results = {
        'x': x_sol[1:],  # exclude initial condition
        'z': z_sol[1:],  # exclude initial condition
        's': s_sol,      # include initial condition
        'obj_val': objective_function(T, p, gamma, delta, c_delivery, eps_delivery, x_sol[1:], z_sol[1:])
    }

    return results

