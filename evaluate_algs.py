import math
import torch
import cvxpy as cp
from functions import load_scenarios_with_flexible
from pald_implementation import (
    make_pald_base_layer,
    make_pald_flex_purchase_layer,
    make_pald_flex_delivery_layer,
)
from paad_implementation import get_alpha
import paad_implementation as pi
from paad_implementation import objective_function as np_objective_function
import argparse
import csv
import statistics
from typing import List, Dict, Any

try:
    import opt_sol  # offline optimal via Gurobi
    _HAS_GUROBI = True
except Exception:
    opt_sol = None
    _HAS_GUROBI = False

# -------------------------
# Config
# -------------------------
T = 96
S = 1.0
K = 10
gamma = 10.0
delta = 5.0
c_delivery = 0.2
eps_delivery = 0.05

# Use the same (no-op) solver_args as in train_pald.py calls for consistency
solver_options = {"solve_method": "ECOS"}  # diffcp still uses SCS internally

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PALD-Fast and PAAD over many instances.")
    parser.add_argument("--num_instances", type=int, default=100, help="Number of instances to evaluate (default: 1)")
    parser.add_argument("--trace", type=str, default="CAISO", help="Trace name (default from file config)")
    parser.add_argument("--month", type=int, default=1, help="Optional month filter")
    return parser.parse_args()

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

def _safe_layer_call(layer, args, fallback=0.0):
    (val,) = layer(*args, solver_args=solver_options)
    return torch.clamp(val, min=0.0)

def base_threshold(w: float, p_min: float, p_max: float, gamma: float, delta: float, c: float, eps: float, T: int, alpha: float, b: float = 1.0) -> float:
    lhs = p_max + 2 * gamma + p_min * c
    inside_exp = (p_max * (1 + c + eps) + 2 * (gamma + delta)) / alpha - (p_max * (1 + eps) + p_min * c + 2 * (gamma + delta) / T)
    return lhs + inside_exp * math.exp(w / (alpha * b))

# New: analytical flexible thresholds (mirror PAAD FlexibleDriver)
def flex_purchase_threshold(w: float, p_min: float, p_max: float, gamma: float, delta: float,
                            c: float, eps: float, T: int, alpha: float, f: float = 1.0) -> float:
    # alpha' from PAAD
    alpha_p = alpha * (1.0 + eps) / (1.0 + c + eps)
    omega = (1.0 + c + eps) / (1.0 + eps)
    lhs = p_max + 2.0 * gamma + p_min * c
    inside = (p_max + 2.0 * gamma) / alpha_p - (p_max + p_min * c + (2.0 * gamma / T) * omega)
    return lhs + inside * math.exp(w / (alpha_p * max(f, 1e-8)))

def flex_delivery_threshold(v: float, p_min: float, p_max: float, gamma: float, delta: float,
                            c: float, eps: float, T: int, alpha: float, f: float = 1.0) -> float:
    alpha_p = alpha * (1.0 + eps) / (1.0 + c + eps)
    omega = (1.0 + c + eps) / (1.0 + eps)
    lhs = p_max * (c + eps) + 2.0 * delta
    inside = (p_max * (c + eps) + 2.0 * delta) / alpha_p - (p_max * (c + eps) + (2.0 * delta / T) * omega)
    return lhs + inside * math.exp(v / (alpha_p * max(f, 1e-8)))

# New: build manually tuned thresholds
def build_tuned_thresholds(K, p_min, p_max, gamma, delta, c, eps, T, alpha):

    # base pieces
    # yb = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    yb = [60 for _ in range(K)]

    # flex purchase
    # yp = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
    yp = [60 for _ in range(K)]

    # flex delivery
    # yd = [25, 25, 25, 25, 25, 25, 25, 25, 25, 25]
    yd = [30 for _ in range(K)]
          
    # enforce non-increasing and pin tails
    for i in range(1, K):
        yb[i] = min(yb[i], yb[i - 1])
        yp[i] = min(yp[i], yp[i - 1])
        yd[i] = min(yd[i], yd[i - 1])
    if K > 0:
        yb[-1] = float(p_min) + 2.0 * gamma
        yp[-1] = float(p_min) + 2.0 * gamma
        yd[-1] = float(p_min) * (c + eps) + 2.0 * delta

    return yb, yp, yd

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

def forward_pald_fast_reuse(models, price_seq, base_seq, flex_seq, Delta_seq,
                             y_base, y_flex_p, y_flex_d, K, gamma, delta,
                             c_delivery, eps_delivery):
    """Same logic as forward_pald_fast but reusing pre-built CVXPy models."""
    base_m, flex_p_m, flex_d_m = models

    x_list, z_list, s_list = [], [], []
    storage_state = 0.0
    x_prev_global = 0.0

    base_drivers = [{"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0}]
    flex_drivers = []

    for t in range(T):
        b_t_val = float(base_seq[t])
        p_t_val = float(price_seq[t])
        # arrivals
        if b_t_val > 0:
            base_drivers.append({"id": 2 * t + 2, "b": b_t_val, "w": 0.0, "prev_decision": 0.0})
        f_arrival = float(flex_seq[t])
        if f_arrival > 0:
            flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": int(Delta_seq[t]),
                                 "w": 0.0, "v": 0.0, "prev_x": 0.0, "prev_z": 0.0})

        prev_purchasing_total = sum(drv["prev_decision"] * drv["b"] for drv in base_drivers)
        prev_purchasing_total += sum(fd["prev_x"] * fd["f"] for fd in flex_drivers)
        purchasing_excess = x_prev_global - prev_purchasing_total

        decisions = []
        # base purchases
        for drv in base_drivers:
            b_i = drv["b"]
            prev_frac = drv["prev_decision"]
            denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
            share = (prev_frac * b_i) / denom if prev_purchasing_total > 0 else 0.0
            pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(b_i, 1e-8)
            w_eff = max(0.0, min(1.0 - 1e-9, float(drv["w"])))
            caps_list = compute_segment_caps(w_eff, K)
            if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                cur_frac = 0.0
            else:
                x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_frac)))
                cur_frac = _solve_base_cvx(base_m, x_prev_clamped, w_eff, p_t_val, y_base, caps_list)
            cur_phys = float(cur_frac) * b_i
            decisions.append(cur_phys)
            drv["prev_decision"] = float(cur_frac)
            drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))

        # flex purchases
        for fd in flex_drivers:
            f_i = fd["f"]
            prev_frac = fd["prev_x"]
            denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
            share = (prev_frac * f_i) / denom if prev_purchasing_total > 0 else 0.0
            pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(f_i, 1e-8)
            w_eff = max(0.0, min(1.0 - 1e-9, float(fd["w"])))
            caps_list = compute_segment_caps(w_eff, K)
            if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                cur_frac = 0.0
            else:
                x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_frac)))
                cur_frac = _solve_flex_purchase_cvx(flex_p_m, x_prev_clamped, w_eff, p_t_val, y_flex_p, caps_list)
            cur_phys = float(cur_frac) * f_i
            decisions.append(cur_phys)
            fd["prev_x"] = float(cur_frac)
            fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))

        x_t = sum(decisions)

        # deliveries
        z_components = [b_t_val]
        deadline_needs = []
        for idx_fd, fd in enumerate(flex_drivers):
            f_i = fd["f"]
            v_prev = float(fd["v"])
            w_prev = float(fd["w"])
            v_eff = max(0.0, min(1.0 - 1e-9, v_prev))
            caps_list = compute_segment_caps(v_eff, K)
            if (1.0 - v_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                cur_frac = 0.0
            else:
                z_prev_clamped = max(0.0, min(1.0 - v_eff, float(fd["prev_z"])))
                coeff = p_t_val * ((c_delivery + eps_delivery) - c_delivery * max(0.0, storage_state))
                cur_frac = _solve_flex_delivery_cvx(flex_d_m, z_prev_clamped, v_eff, coeff, y_flex_d, caps_list)
            if T and (t >= max(0, int(fd["delta"]) - 1)):
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
        x_list.append(x_t)
        z_list.append(z_t)
        s_list.append(storage_state)

    return x_list, z_list, s_list

def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    vs = sorted(values)
    def pct(p): return vs[int(p * (len(vs) - 1))]
    return {
        "mean": float(sum(vs) / len(vs)),
        "median": float(statistics.median(vs)),
        "p10": pct(0.10),
        "p25": pct(0.25),
        "p75": pct(0.75),
        "p95": pct(0.95),
        "min": vs[0],
        "max": vs[-1],
    }

def evaluate_many(price_all, base_all, flex_all, Delta_all, p_min, p_max, y_base, y_flex_p, y_flex_d):
    num_instances = len(price_all)
    print(f"Evaluating {num_instances} instances...")

    models = _build_models_once(K, gamma, delta)

    pald_costs = []
    paad_costs = []
    opt_costs = []
    pald_delivered = []
    paad_delivered = []
    opt_delivered = []

    rows: List[Dict[str, Any]] = []

    for idx in range(num_instances):
        p_seq = price_all[idx]
        b_seq = base_all[idx]
        f_seq = flex_all[idx]
        D_seq = Delta_all[idx]

        # PALD-Fast
        pald_x, pald_z, _ = forward_pald_fast_reuse(
            models, p_seq, b_seq, f_seq, D_seq,
            y_base, y_flex_p, y_flex_d, K, gamma, delta, c_delivery, eps_delivery
        )
        pald_cost = np_objective_function(T, p_seq, gamma, delta, c_delivery, eps_delivery, pald_x, pald_z)

        # PAAD
        paad_res = pi.paad_algorithm(T, p_seq, gamma, delta,
                                     c_delivery, eps_delivery,
                                     p_min, p_max, S, b_seq, f_seq, D_seq)
        paad_x = paad_res["x"]
        paad_z = paad_res["z"]
        paad_cost = np_objective_function(T, p_seq, gamma, delta, c_delivery, eps_delivery, paad_x, paad_z)

        pald_costs.append(pald_cost)
        paad_costs.append(paad_cost)
        pald_delivered.append(float(sum(pald_z)))
        paad_delivered.append(float(sum(paad_z)))

        row = {
            "instance": idx,
            "pald_cost": pald_cost,
            "paad_cost": paad_cost,
            "pald_delivered": float(sum(pald_z)),
            "paad_delivered": float(sum(paad_z)),
        }

        # OPT (optional)
        if _HAS_GUROBI and opt_sol is not None:
            try:
                status, results = opt_sol.optimal_solution(
                    T, p_seq, gamma, delta, c_delivery, eps_delivery, S, b_seq, f_seq, D_seq
                )
                if status == "Optimal" and results is not None:
                    oc = np_objective_function(T, p_seq, gamma, delta, c_delivery, eps_delivery,
                                               results["x"], results["z"])
                    opt_costs.append(oc)
                    delivered_opt = float(sum(results["z"]))
                    opt_delivered.append(delivered_opt)
                    row["opt_cost"] = oc
                    row["opt_delivered"] = delivered_opt
                    row["pald_over_opt"] = pald_cost / oc if oc > 0 else None
                    row["paad_over_opt"] = paad_cost / oc if oc > 0 else None
                else:
                    row["opt_cost"] = None
            except Exception as e:
                row["opt_cost"] = None
        rows.append(row)

    # Aggregates
    print("\n=== Aggregate Results ===")
    def print_summary(label, vals):
        if not vals:
            print(f"{label}: (none)")
            return
        s = summarize(vals)
        print(f"{label}:    mean={s['mean']:.4f}    median={s['median']:.4f}    p95={s['p95']:.4f}    min={s['min']:.4f}")

    # print_summary("PALD-Fast cost", pald_costs)
    # print_summary("PAAD cost", paad_costs)
    if opt_costs:
        # print_summary("OPT cost", opt_costs)
        ratios_pald = [r["pald_over_opt"] for r in rows if r.get("pald_over_opt")]
        ratios_paad = [r["paad_over_opt"] for r in rows if r.get("paad_over_opt")]
        print_summary("PALD/OPT", ratios_pald)
        print_summary("PAAD/OPT", ratios_paad)

    # print_summary("Delivered PALD-Fast", pald_delivered)
    # print_summary("Delivered PAAD", paad_delivered)
    # if opt_delivered:
    #     print_summary("Delivered OPT", opt_delivered)

    return rows

def main():
    args = parse_args()

    print(f"Evaluating {args.num_instances} instances (trace={args.trace}, month={args.month})...")
    price_all, base_all, flex_all, Delta_all, p_min, p_max = load_scenarios_with_flexible(
        args.num_instances, T, args.trace, month=args.month
    )

    # Thresholds (analytical) – same for all instances
    alpha = float(get_alpha(float(p_min), float(p_max), c_delivery, eps_delivery, T, gamma, delta))
    w_grid = [(i + 0.5) / K for i in range(K)]

    y_base = [base_threshold(w, float(p_min), float(p_max), gamma, delta,
                             c_delivery, eps_delivery, T, alpha, b=1.0) for w in w_grid]
    for i in range(1, K):
        y_base[i] = min(y_base[i], y_base[i-1])
    if K > 0:
        y_base[-1] = float(p_min) + 2.0 * gamma

    y_flex_p = [flex_purchase_threshold(w, float(p_min), float(p_max), gamma, delta,
                                        c_delivery, eps_delivery, T, alpha, f=1.0) for w in w_grid]
    for i in range(1, K):
        y_flex_p[i] = min(y_flex_p[i], y_flex_p[i-1])
    if K > 0:
        y_flex_p[-1] = float(p_min) + 2.0 * gamma

    y_flex_d = [flex_delivery_threshold(v, float(p_min), float(p_max), gamma, delta,
                                        c_delivery, eps_delivery, T, alpha, f=1.0) for v in w_grid]
    for i in range(1, K):
        y_flex_d[i] = min(y_flex_d[i], y_flex_d[i-1])
    if K > 0:
        y_flex_d[-1] = float(p_min) * (c_delivery + eps_delivery) + 2.0 * delta

    evaluate_many(price_all, base_all, flex_all, Delta_all, p_min, p_max, y_base, y_flex_p, y_flex_d)

if __name__ == "__main__":
    main()

