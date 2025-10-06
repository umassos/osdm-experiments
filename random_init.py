
import numpy as np
import torch
import cvxpy as cp
from robust_projection import project_y_robust, project_y_flex_robust
from paad_implementation import objective_function as np_objective_function
from functions import load_scenarios_with_flexible
from paad_implementation import get_alpha
import os 
import pickle
import opt_sol
from tqdm import tqdm

# -------------------------
# Helpers reused from train_pald.py (no training)
# -------------------------
def compute_segment_caps(w_prev: float, K: int):
    w = max(0.0, min(1.0, float(w_prev)))
    if 1.0 - w <= 1e-9:
        return [0.0] * K
    caps = []
    for i in range(K):
        left = i / K
        right = (i + 1) / K
        cap = max(0.0, right - max(left, w))
        caps.append(cap)
    return caps

# -------------------------
# Forward PALD (fast CVXPY variant, no cvxpylayers)
# -------------------------
def _build_pald_base_cvx(K, gamma):
    x_parts = cp.Variable(K, nonneg=True)
    x_total = cp.Variable(nonneg=True)
    # Parameters
    x_prev = cp.Parameter(nonneg=True)
    w_prev = cp.Parameter(nonneg=True)
    p_t = cp.Parameter()
    y_vec = cp.Parameter(K)
    caps = cp.Parameter(K, nonneg=True)
    constraints = [
        x_parts >= 0,
        x_parts <= caps,
        x_total == cp.sum(x_parts),
        x_total <= 1 - w_prev,
    ]
    obj = cp.Minimize(p_t * x_total + gamma * cp.abs(x_total - x_prev) + gamma * cp.abs(x_total) - y_vec @ x_parts)
    prob = cp.Problem(obj, constraints)
    return {"prob": prob, "x_prev": x_prev, "w_prev": w_prev, "p_t": p_t, "y_vec": y_vec, "caps": caps, "x_total": x_total}

def _build_pald_flex_purchase_cvx(K, gamma):
    # identical to base, separate instance for clarity
    return _build_pald_base_cvx(K, gamma)

def _build_pald_flex_delivery_cvx(K, delta):
    z_parts = cp.Variable(K, nonneg=True)
    z_total = cp.Variable(nonneg=True)
    # Parameters
    z_prev = cp.Parameter(nonneg=True)
    v_prev = cp.Parameter(nonneg=True)
    coeff = cp.Parameter()
    y_vec = cp.Parameter(K)
    caps = cp.Parameter(K, nonneg=True)
    constraints = [
        z_parts >= 0,
        z_parts <= caps,
        z_total == cp.sum(z_parts),
        z_total <= 1 - v_prev,
    ]
    obj = cp.Minimize(coeff * z_total + delta * cp.abs(z_total - z_prev) + delta * cp.abs(z_total) - y_vec @ z_parts)
    prob = cp.Problem(obj, constraints)
    return {"prob": prob, "z_prev": z_prev, "v_prev": v_prev, "coeff": coeff, "y_vec": y_vec, "caps": caps, "z_total": z_total}

_CLARABEL_KW = dict(solver=cp.CLARABEL, verbose=False, warm_start=True)

def _solve_base_cvx(model, x_prev, w_prev, p_t, y_vec, caps):
    if (1.0 - w_prev) <= 1e-12 or (sum(caps) <= 1e-12):
        return 0.0
    model["x_prev"].value = max(0.0, float(x_prev))
    model["w_prev"].value = max(0.0, min(1.0, float(w_prev)))
    model["p_t"].value = float(p_t)
    model["y_vec"].value = list(map(float, y_vec))
    model["caps"].value = list(map(float, caps))
    try:
        model["prob"].solve(**_CLARABEL_KW)
        val = model["x_total"].value
        return max(0.0, float(val) if val is not None else 0.0)
    except Exception:
        return 0.0

def _solve_flex_purchase_cvx(model, x_prev, w_prev, p_t, y_vec, caps):
    return _solve_base_cvx(model, x_prev, w_prev, p_t, y_vec, caps)

def _solve_flex_delivery_cvx(model, z_prev, v_prev, coeff, y_vec, caps):
    if (1.0 - v_prev) <= 1e-12 or (sum(caps) <= 1e-12):
        return 0.0
    model["z_prev"].value = max(0.0, float(z_prev))
    model["v_prev"].value = max(0.0, min(1.0, float(v_prev)))
    model["coeff"].value = float(coeff)
    model["y_vec"].value = list(map(float, y_vec))
    model["caps"].value = list(map(float, caps))
    try:
        model["prob"].solve(**_CLARABEL_KW)
        val = model["z_total"].value
        return max(0.0, float(val) if val is not None else 0.0)
    except Exception:
        return 0.0

def _build_models_once(K, gamma, delta):
    """Helper to build CVXPy models once and reuse (minor speedup)."""
    base_m = _build_pald_base_cvx(K, gamma)
    flex_p_m = _build_pald_flex_purchase_cvx(K, gamma)
    flex_d_m = _build_pald_flex_delivery_cvx(K, delta)
    return base_m, flex_p_m, flex_d_m

def random_base_vector(K, T, price_min, price_max, p_min, p_max, gamma, delta, c_delivery, eps_delivery, beta):
    y_random = torch.rand(K) * (float(price_max) - float(price_min)) + float(price_min)
    y_random, _ = torch.sort(y_random, descending=True)
    y_random[-1] = float(p_min) + 2.0 * float(gamma)

    # project to ensure robustness
    y_proj = project_y_robust(y_random, K, float(p_min), float(p_max), float(gamma), float(delta), float(c_delivery), float(eps_delivery), int(T), beta=beta)
    if y_proj is not None:
        for i in range(K):
            y_random[i] = torch.tensor(float(y_proj[i]))
    if K > 0:
        y_random[-1] = torch.tensor(float(p_min) + 2.0 * gamma)
    
    return y_random

def random_flex_vectors(K, T, price_min, price_max, p_min, p_max, gamma, delta, c_delivery, eps_delivery, beta):
    y_flex_p_random = torch.rand(K) * (float(price_max) - float(price_min)) + float(price_min)
    y_flex_p_random, _ = torch.sort(y_flex_p_random, descending=True)
    y_flex_p_random[-1] = float(price_min) + 2.0 * float(gamma)

    y_flex_d_random = torch.rand(K) * (float(price_max) * (c_delivery + eps_delivery) - float(price_min) * (c_delivery + eps_delivery)) + float(price_min) * (c_delivery + eps_delivery)
    y_flex_d_random, _ = torch.sort(y_flex_d_random, descending=True)
    y_flex_d_random[-1] = float(price_min) * (c_delivery + eps_delivery) + 2.0 * float(delta)

    # project to ensure robustness
    phi_proj, psi_proj = project_y_flex_robust(
        y_flex_p_random, y_flex_d_random, K,
        float(p_min), float(p_max),
        float(gamma), float(delta),
        float(c_delivery), float(eps_delivery),
        int(T), beta=beta
    )
    for i in range(K):
        y_flex_p_random[i] = torch.tensor(float(phi_proj[i]))
        y_flex_d_random[i] = torch.tensor(float(psi_proj[i]))
    # clamp to max ranges (safety)
    for i in range(K):
        y_flex_p_random[i] = torch.clamp(y_flex_p_random[i], max=float(p_max))
        y_flex_d_random[i] = torch.clamp(y_flex_d_random[i], max=float(p_max) * (c_delivery + eps_delivery))
    
    if K > 0:
        y_flex_p_random[-1] = torch.tensor(float(price_min) + 2.0 * gamma)
        y_flex_d_random[-1] = torch.tensor(float(price_min) * (c_delivery + eps_delivery) + 2.0 * delta)

    return y_flex_p_random, y_flex_d_random

def forward_pald_fast_random(models, price_seq, base_seq, flex_seq, Delta_seq,
                             S, K, T, gamma, delta, c_delivery, eps_delivery, p_min, p_max, beta,
                             y_base_vectors=None, y_flex_p_vectors=None, y_flex_d_vectors=None):
    """Same logic as forward_pald_fast but reusing pre-built CVXPy models."""
    base_m, flex_p_m, flex_d_m = models

    price_max = max(price_seq)
    price_min = min(price_seq)

    x_hist = []
    z_hist = []
    storage_state = 0.0
    x_prev_global = 0.0

    if y_base_vectors is None:
        y_base_vectors = {}
    if y_flex_p_vectors is None:
        y_flex_p_vectors = {}
    if y_flex_d_vectors is None:
        y_flex_d_vectors = {}

    # generate a random vector for this initial base driver
    y_base = random_base_vector(K, T, p_min, p_max, p_min, p_max, gamma, delta, c_delivery, eps_delivery, beta=beta)
    base_drivers = [{"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0, "y": y_base}]
    y_base_vectors[0] = y_base
    flex_drivers = []
    

    for t in range(T):
        b_t_val = float(base_seq[t])
        p_t_val = float(price_seq[t])

        # arrivals
        if b_t_val > 0:
            # generate a random vector for this new base driver
            y_base = random_base_vector(K, T, p_min, p_max, p_min, p_max, gamma, delta, c_delivery, eps_delivery, beta=beta)
            base_drivers.append({"id": 2 * t + 2, "b": b_t_val, "w": 0.0, "prev_decision": 0.0, "y": y_base})
            y_base_vectors[2 * t + 2] = y_base
        f_arrival = float(flex_seq[t])
        if f_arrival > 0:
            # generate random vectors for this new flex driver
            y_flex_p, y_flex_d = random_flex_vectors(K, T, p_min, p_max, p_min, p_max, gamma, delta, c_delivery, eps_delivery, beta=beta)
            y_flex_p_vectors[2 * t + 1] = y_flex_p
            y_flex_d_vectors[2 * t + 1] = y_flex_d
            flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": int(Delta_seq[t]),
                                 "w": 0.0, "v": 0.0, "prev_x": 0.0, "prev_z": 0.0, "yp": y_flex_p, "yd": y_flex_d})

        prev_purchasing_total = sum(drv["prev_decision"] * drv["b"] for drv in base_drivers)
        prev_purchasing_total += sum(fd["prev_x"] * fd["f"] for fd in flex_drivers)
        purchasing_excess = x_prev_global - prev_purchasing_total

        # deliveries
        z_components = [b_t_val]
        deadline_needs = []

        for idx_fd, fd in enumerate(flex_drivers):
            f_i = fd["f"]
            v_prev = float(fd["v"])
            w_prev = float(fd["w"])
            v_eff = max(0.0, min(1.0 - 1e-9, v_prev))
            y_flex_d = fd["yd"]
            caps_list = compute_segment_caps(v_eff, K)
            if (1.0 - v_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                cur_frac = 0.0
            else:
                z_prev_clamped = max(0.0, min(1.0 - v_eff, float(fd["prev_z"])))
                coeff = p_t_val * ((c_delivery + eps_delivery) - c_delivery * max(0.0, storage_state))
                cur_frac = _solve_flex_delivery_cvx(flex_d_m, z_prev_clamped, v_eff, coeff, y_flex_d, caps_list)
            if T and (t >= max(0, int(fd["delta"])-1)):
                cur_frac = max(0.0, 1.0 - v_prev)
                avail_frac = max(0.0, w_prev - v_prev)
                need_frac = max(0.0, cur_frac - avail_frac)
                need_phys = need_frac * f_i
                if need_phys > 0:
                    deadline_needs.append((idx_fd, need_phys))
            else:
                cur_frac = min(cur_frac, max(0.0, w_prev - v_prev))
            cur_phys = float(cur_frac) * f_i
            z_components.append(cur_phys)
            fd["prev_z"] = float(cur_frac)

        z_t = sum(z_components)

        buy_cap = S - storage_state + z_t

        decisions = []
        # flex purchases
        for fd in flex_drivers:
            if buy_cap <= 1e-9:
                # no more buying possible, force zero decision
                decisions.append(0.0)
                fd["prev_x"] = 0.0
                fd["w"] = float(fd["w"])  # no change
                continue
            
            f_i = fd["f"]
            prev_frac = fd["prev_x"]
            denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
            share = (prev_frac * f_i) / denom if prev_purchasing_total > 0 else 0.0
            pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(f_i, 1e-8)
            w_eff = max(0.0, min(1.0 - 1e-9, float(fd["w"])))
            y_flex_p = fd["yp"]
            caps_list = compute_segment_caps(w_eff, K)
            if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                cur_frac = 0.0
            else:
                x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_frac)))
                cur_frac = _solve_flex_purchase_cvx(flex_p_m, x_prev_clamped, w_eff, p_t_val, y_flex_p, caps_list)
            if T and (t >= max(0, int(fd["delta"]))):
                cur_frac = max(0.0, 1.0 - w_prev)
            cur_phys = float(cur_frac) * f_i
            decisions.append(cur_phys)
            fd["prev_x"] = float(cur_frac)
            fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))

        # base purchases
        for drv in base_drivers:
            if buy_cap <= 1e-9:
                # no more buying possible, force zero decision
                decisions.append(0.0)
                drv["prev_decision"] = 0.0
                drv["w"] = float(drv["w"])  # no change
                continue
                
            b_i = drv["b"]
            prev_frac = drv["prev_decision"]
            denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
            share = (prev_frac * b_i) / denom if prev_purchasing_total > 0 else 0.0
            pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(b_i, 1e-8)
            w_eff = max(0.0, min(1.0 - 1e-9, float(drv["w"])))
            y_base = drv["y"]
            
            caps_list = compute_segment_caps(w_eff, K)
            if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                cur_frac = 0.0
            else:
                x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_frac)))
                cur_frac = _solve_base_cvx(base_m, x_prev_clamped, w_eff, p_t_val, y_base, caps_list)
            
            cur_phys = float(cur_frac) * b_i

            # check if this decision exceeds the remaining buy cap
            if cur_phys - buy_cap > 1e-9:
                cur_phys = buy_cap
                cur_frac = cur_phys / max(b_i, 1e-8)
                buy_cap = 0.0

            decisions.append(cur_phys)
            drv["prev_decision"] = float(cur_frac)
            drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))


        x_t = sum(decisions)

        # Inventory feasibility + same-slot top-up
        x_required = max(0.0, z_t - storage_state)
        if x_t + 1e-12 < x_required:
            extra_phys = x_required - x_t
            total_need = sum(n for _, n in deadline_needs)
            if total_need > 1e-12:
                for idx_fd, need_phys in deadline_needs:
                    alloc_phys = extra_phys * (need_phys / total_need)
                    fd = flex_drivers[idx_fd]
                    inc_frac = min(1.0 - float(fd["w"]), alloc_phys / max(fd["f"], 1e-8))
                    if inc_frac > 0:
                        fd["prev_x"] += inc_frac
                        fd["w"] = float(min(1.0, fd["w"] + inc_frac))
            x_t += extra_phys

        for fd in flex_drivers:
            fd["v"] = float(min(1.0, fd["v"] + fd["prev_z"]))

        storage_state = storage_state + x_t - z_t
        x_prev_global = x_t
        x_hist.append(x_t)
        z_hist.append(z_t)

    # ensure that the sum of z_hist meets the demand
    total_base = sum(float(drv["b"]) for drv in base_drivers)
    total_flex = sum(float(fd["f"]) for fd in flex_drivers)
    total_demand = total_base + total_flex
    total_z = sum(z_hist)
    if abs(total_demand - total_z) > 1e-2:
        print(f"[check:] Total demand: {total_demand}, total z: {total_z}, total x: {sum(x_hist)}")

    return x_hist, z_hist, y_base_vectors, y_flex_p_vectors, y_flex_d_vectors


def initialize_pald_thresholds(K, price_all, base_all, flex_all, Delta_all, 
                            gamma, delta, c_delivery, eps_delivery, p_min, p_max, beta,
                               opt_costs, rounds=500):
    """
    Initialize random thresholds for PALD with integral-aware allocation.

    Args:
        K (int): Number of segments.
        price_all (list of list of float): List of price sequences for each scenario.
        base_all (list of list of float): List of base demand sequences for each scenario.
        flex_all (list of list of float): List of flexible demand sequences for each scenario.
        Delta_all (list of list of float): List of deadline sequences for each scenario.
        opt_costs (list of float): List of optimal costs for each scenario.
        rounds (int): Number of random initialization rounds.
    Returns:
        base_targets (list of float): Initial thresholds for base purchase.
        flexp_targets (list of float): Initial thresholds for flex purchase.
        flexd_targets (list of float): Initial thresholds for flex delivery.
    """
    # for each instance, we run forward_pald_fast_random and compute cost 500 times, save the best
    best_costs = [float('inf')] * len(price_all)
    best_base_targets = [None] * len(price_all)
    best_flexp_targets = [None] * len(price_all)
    best_flexd_targets = [None] * len(price_all)
    num_instances = len(price_all)
    models = _build_models_once(K, gamma, delta)

    # use TQDM to show progress
    for i in tqdm(range(num_instances), desc="instances"):
        best_cost = float('inf')
        best_base_vectors = None
        best_flexp_vectors = None
        best_flexd_vectors = None

        price_seq = price_all[i]
        base_seq = base_all[i]
        flex_seq = flex_all[i]
        Delta_seq = Delta_all[i]
        T = len(price_seq)
        for r in range(rounds):
            x_list, z_list, y_base_vectors, y_flex_p_vectors, y_flex_d_vectors = forward_pald_fast_random(
                models, price_seq, base_seq, flex_seq, Delta_seq,
                S=0.0, K=K, T=T, gamma=gamma, delta=delta,
                c_delivery=c_delivery, eps_delivery=eps_delivery,
                p_min=p_min, p_max=p_max, beta=beta
            )
            cost = np_objective_function(T, price_seq, gamma, delta, c_delivery, eps_delivery, x_list, z_list)
            if cost < best_cost:
                best_cost = cost
                best_base_vectors = y_base_vectors
                best_flexp_vectors = y_flex_p_vectors
                best_flexd_vectors = y_flex_d_vectors
                # print competitive ratio for this round
                if opt_costs[i] > 0:
                    cr = cost / opt_costs[i]
                    print(f"[update:] Instance {i}, round {r}, cost: {cost:.2f}, OPT: {opt_costs[i]:.2f}, CR: {cr:.4f}")
        best_costs[i] = best_cost
        best_base_targets[i] = best_base_vectors
        best_flexp_targets[i] = best_flexp_vectors
        best_flexd_targets[i] = best_flexd_vectors

    avg_cr = sum(best_costs[i] / opt_costs[i] for i in range(num_instances) if opt_costs[i] > 0) / num_instances
    print(f"Average competitive ratio over {num_instances} instances: {avg_cr:.4f}")

    return best_base_targets, best_flexp_targets, best_flexd_targets

if __name__ == "__main__":
    total_instances = 10
    T=48
    c_delivery = 0.2
    eps_delivery = 0.05
    gamma = 10.0
    delta = 5.0
    S = 1.0
    trace = "CAISO"
    month = 99  # 1-12 for monthly data, 99 for all
    price_all, base_all, flex_all, Delta_all, p_min, p_max = load_scenarios_with_flexible(total_instances, T, trace, month=month, saved=False)

    alpha = float(get_alpha(float(p_min), float(p_max), c_delivery, eps_delivery, 96, gamma, delta))
    # print(f"Computed alpha for analytical thresholds: {alpha}")
    beta = 100

    # ---------------------------------------
    # Precompute OPT costs for competitive-ratio loss
    # ---------------------------------------
    def precompute_opt_costs_flex(price_instances, base_instances, flex_instances, Delta_instances,
                                T, gamma, delta, c, eps, S):
        """
        Returns a list (len = total_instances) of OPT objective values (floats) or None per instance.
        """

        opt_costs = []

        # the total demand should match exactly, we can use this to verify our cache
        total_demands = []
        for b_seq, f_seq in zip(base_instances, flex_instances):
            total_demand = sum(b_seq) + sum(f_seq)
            total_demands.append(total_demand)

        # if os.path.exists(f"opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl"):
        #     with open(f"opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl", "rb") as f:
        #         opt_costs, total_demands_saved = pickle.load(f)
        #     print(f"Loaded precomputed OPT costs for flexible demand from opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl")

        #     # verify that the saved total demands match
        #     if total_demands != total_demands_saved:
        #         print("Warning: Total demands do not match the saved values, recomputing OPT costs.") # force recomputation
        #         opt_costs = []  # force recomputation
        #     else:
        #         return opt_costs, total_demands

        # use TQDM for progress bar
        for p_seq, b_seq, f_seq, dlt in tqdm(zip(price_instances, base_instances, flex_instances, Delta_instances)):
            try:
                status, results = opt_sol.optimal_solution(T, p_seq, gamma, delta, c, eps, S, b_seq, f_seq, dlt)
                if status == "Optimal" and results is not None:
                    opt_cost = np_objective_function(T, p_seq, gamma, delta, c, eps, results['x'], results['z'])
                else:
                    opt_cost = None
            except Exception:
                opt_cost = None
            opt_costs.append(opt_cost)
        
        # save the computed OPT costs for future use
        # first ensure the directory exists
        os.makedirs("opt_sols", exist_ok=True)
        with open(f"opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl", "wb") as f:
            pickle.dump((opt_costs, total_demands), f)
        
        return opt_costs, total_demands

    print("Precomputing OPT costs for competitive-ratio loss...")
    opt_costs_all, total_demands_all = precompute_opt_costs_flex(price_all, base_all, flex_all, Delta_all, T, gamma, delta, c_delivery, eps_delivery, S)
    num_opt_ok = sum(1 for v in (opt_costs_all or []) if (v is not None and v > 1e-6))
    print(f"OPT costs available for {num_opt_ok}/{total_instances} instances.")

    print("Initializing random thresholds for PALD...")
    base_targets, flexp_targets, flexd_targets = initialize_pald_thresholds(
        K=11,
        price_all=price_all,
        base_all=base_all,
        flex_all=flex_all,
        Delta_all=Delta_all,
        gamma=gamma,
        delta=delta,
        c_delivery=c_delivery,
        eps_delivery=eps_delivery,
        p_min=p_min,
        p_max=p_max,
        beta=beta,
        opt_costs=opt_costs_all,
        rounds=50
    )