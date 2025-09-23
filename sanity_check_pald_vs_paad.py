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
trace = "CAISO"

# Use the same (no-op) solver_args as in train_pald.py calls for consistency
solver_options = {"solve_method": "ECOS"}  # diffcp still uses SCS internally

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
# Forward PALD with given thresholds
# -------------------------
def forward_pald(price_seq, base_seq, flex_seq, Delta_seq, y_base, y_flex_p, y_flex_d, K, gamma, delta, c_delivery, eps_delivery):
    pald_base_layer = make_pald_base_layer(K, gamma)
    pald_flex_purchase_layer = make_pald_flex_purchase_layer(K, gamma)
    pald_flex_delivery_layer = make_pald_flex_delivery_layer(K, delta, c_delivery, eps_delivery)

    x_list, z_list, s_list = [], [], []
    storage_state = 0.0
    x_prev_global = torch.tensor(0.0)

    # thresholds as tensors
    y = torch.tensor(y_base, dtype=torch.float32)
    yp = torch.tensor(y_flex_p, dtype=torch.float32)
    yd = torch.tensor(y_flex_d, dtype=torch.float32)

    base_drivers = [{"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0}]
    flex_drivers = []

    with torch.no_grad():
        for t in range(T):
            b_t_val = float(base_seq[t])
            p_t_val = float(price_seq[t])
            # arrivals
            if b_t_val > 0:
                base_drivers.append({"id": 2 * t + 2, "b": b_t_val, "w": 0.0, "prev_decision": 0.0})
            f_arrival = float(flex_seq[t])
            if f_arrival > 0:
                flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": int(Delta_seq[t]), "w": 0.0, "v": 0.0, "prev_x": 0.0, "prev_z": 0.0})

            prev_purchasing_total = sum(drv["prev_decision"] * drv["b"] for drv in base_drivers)
            prev_purchasing_total += sum(fd["prev_x"] * fd["f"] for fd in flex_drivers)
            purchasing_excess = x_prev_global.item() - prev_purchasing_total

            decisions = []
            # base purchase
            for drv in base_drivers:
                b_i = drv["b"]
                prev_frac = drv["prev_decision"]
                denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
                share = (prev_frac * b_i) / denom if prev_purchasing_total > 0 else 0.0
                pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(b_i, 1e-8)
                w_prev_frac = float(drv["w"])
                w_eff = max(0.0, min(1.0 - 1e-9, w_prev_frac))
                caps_list = compute_segment_caps(w_eff, K)
                if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                    cur_frac_decision = torch.tensor(0.0, dtype=torch.float32)
                else:
                    x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_frac)))
                    x_prev_frac_t = torch.tensor(x_prev_clamped, dtype=torch.float32)
                    w_prev_frac_t = torch.tensor(w_eff, dtype=torch.float32)
                    p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                    caps_t = torch.tensor(caps_list, dtype=torch.float32)
                    cur_frac_decision = _safe_layer_call(
                        pald_base_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y, caps_t), fallback=(1.0 - w_eff)
                    )
                cur_phys_decision = float(cur_frac_decision.item() * b_i)
                decisions.append(cur_phys_decision)
                drv["prev_decision"] = float(cur_frac_decision.item())
                drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))

            # flex purchase
            for fd in flex_drivers:
                f_i = fd["f"]
                prev_frac = fd["prev_x"]
                denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
                share = (prev_frac * f_i) / denom if prev_purchasing_total > 0 else 0.0
                pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(f_i, 1e-8)
                w_prev_frac = float(fd["w"])
                w_eff = max(0.0, min(1.0 - 1e-9, w_prev_frac))
                caps_list = compute_segment_caps(w_eff, K)
                if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                    cur_frac_x = torch.tensor(0.0, dtype=torch.float32)
                else:
                    x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_frac)))
                    x_prev_frac_t = torch.tensor(x_prev_clamped, dtype=torch.float32)
                    w_prev_frac_t = torch.tensor(w_eff, dtype=torch.float32)
                    p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                    caps_t = torch.tensor(caps_list, dtype=torch.float32)
                    cur_frac_x = _safe_layer_call(
                        pald_flex_purchase_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, yp, caps_t), fallback=(1.0 - w_eff)
                    )
                cur_phys_x = float(cur_frac_x.item() * f_i)
                decisions.append(cur_phys_x)
                fd["prev_x"] = float(cur_frac_x.item())
                fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))

            # Aggregate initial purchases
            x_t = sum(decisions)

            # deliveries
            z_components = [b_t_val]
            # Track deadline purchase needs to attribute same-slot top-up
            deadline_needs = []  # list of (fd_index, need_phys)
            for idx_fd, fd in enumerate(flex_drivers):
                f_i = fd["f"]
                v_prev = float(fd["v"])
                w_prev = float(fd["w"])
                v_eff = max(0.0, min(1.0 - 1e-9, v_prev))
                caps_list = compute_segment_caps(v_eff, K)
                if (1.0 - v_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                    cur_frac_z = torch.tensor(0.0, dtype=torch.float32)
                else:
                    z_prev_clamped = max(0.0, min(1.0 - v_eff, float(fd["prev_z"])))
                    z_prev_frac_t = torch.tensor(z_prev_clamped, dtype=torch.float32)
                    v_prev_frac_t = torch.tensor(v_eff, dtype=torch.float32)
                    caps_t = torch.tensor(caps_list, dtype=torch.float32)
                    coeff_t = torch.tensor(
                        p_t_val * ((c_delivery + eps_delivery) - c_delivery * float(max(0.0, storage_state))),
                        dtype=torch.float32,
                    )
                    cur_frac_z = _safe_layer_call(
                        pald_flex_delivery_layer, (z_prev_frac_t, v_prev_frac_t, coeff_t, yd, caps_t), fallback=(1.0 - v_eff)
                    )
                # Enforce deadline
                if t >= max(0, int(fd["delta"]) - 1):
                    # Force delivery of the remaining fraction
                    cur_frac_z = torch.tensor(max(0.0, 1.0 - v_prev), dtype=torch.float32)
                    # Compute additional purchase needed for this driver this slot (in physical units)
                    avail_frac = max(0.0, w_prev - v_prev)
                    need_frac = max(0.0, float(cur_frac_z.item()) - avail_frac)
                    need_phys = need_frac * f_i
                    if need_phys > 0:
                        deadline_needs.append((idx_fd, need_phys))
                else:
                    # For non-deadline, cap by purchased remainder
                    cur_frac_z = torch.clamp(cur_frac_z, max=max(0.0, w_prev - v_prev))

                cur_phys_z = float(cur_frac_z.item() * f_i)
                z_components.append(cur_phys_z)
                fd["prev_z"] = float(cur_frac_z.item())
                # v will be updated after possible purchase top-up attribution

            z_t = sum(z_components)

            # inventory feasibility and same-slot purchase top-up attribution
            x_required = max(0.0, z_t - storage_state)
            if x_t + 1e-12 < x_required:
                extra_phys = x_required - x_t
                # Attribute extra purchases to deadline jobs that need it to keep v ≤ w
                total_need = sum(need for _, need in deadline_needs)
                if total_need > 1e-12:
                    for idx_fd, need_phys in deadline_needs:
                        alloc_phys = extra_phys * (need_phys / total_need)
                        fd = flex_drivers[idx_fd]
                        # convert to fractional increment for this driver and clip to remaining capacity
                        inc_frac = min(1.0 - float(fd["w"]), alloc_phys / max(fd["f"], 1e-8))
                        if inc_frac > 0:
                            fd["prev_x"] += inc_frac
                            fd["w"] = float(min(1.0, fd["w"] + inc_frac))
                x_t = x_t + extra_phys  # apply global top-up

            # Now update v after purchases (so v ≤ w holds)
            for fd in flex_drivers:
                fd["v"] = float(min(1.0, fd["v"] + fd["prev_z"]))

            # finalize inventory and histories
            storage_state = storage_state + x_t - z_t
            x_prev_global = torch.tensor(x_t)
            x_list.append(x_t)
            z_list.append(z_t)
            s_list.append(storage_state)

    return x_list, z_list, s_list

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

def forward_pald_fast(price_seq, base_seq, flex_seq, Delta_seq, y_base, y_flex_p, y_flex_d, K, gamma, delta, c_delivery, eps_delivery):
    # Build once; update Parameters each call
    base_m = _build_pald_base_cvx(K, gamma)
    flex_p_m = _build_pald_flex_purchase_cvx(K, gamma)
    flex_d_m = _build_pald_flex_delivery_cvx(K, delta)

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
            flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": int(Delta_seq[t]), "w": 0.0, "v": 0.0, "prev_x": 0.0, "prev_z": 0.0})

        prev_purchasing_total = sum(drv["prev_decision"] * drv["b"] for drv in base_drivers)
        prev_purchasing_total += sum(fd["prev_x"] * fd["f"] for fd in flex_drivers)
        purchasing_excess = x_prev_global - prev_purchasing_total

        # base purchases
        decisions = []
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
        deadline_needs = []  # (idx_fd, need_phys)
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
            # Enforce deadline: deliver all remaining
            if t >= max(0, int(fd["delta"]) - 1):
                cur_frac = max(0.0, 1.0 - v_prev)
                avail_frac = max(0.0, w_prev - v_prev)
                need_frac = max(0.0, cur_frac - avail_frac)
                need_phys = need_frac * f_i
                if need_phys > 0:
                    deadline_needs.append((idx_fd, need_phys))
            else:
                # Before deadline, cannot exceed purchased remainder
                cur_frac = min(cur_frac, max(0.0, w_prev - v_prev))
            cur_phys = float(cur_frac) * f_i
            z_components.append(cur_phys)
            fd["prev_z"] = float(cur_frac)
            # v is updated after handling top-up attribution

        z_t = sum(z_components)

        # Inventory feasibility and same-slot top-up attribution for deadlines
        x_required = max(0.0, z_t - storage_state)
        if x_t + 1e-12 < x_required:
            extra_phys = x_required - x_t
            total_need = sum(need for _, need in deadline_needs)
            if total_need > 1e-12:
                for idx_fd, need_phys in deadline_needs:
                    alloc_phys = extra_phys * (need_phys / total_need)
                    fd = flex_drivers[idx_fd]
                    inc_frac = min(1.0 - float(fd["w"]), alloc_phys / max(fd["f"], 1e-8))
                    if inc_frac > 0:
                        fd["prev_x"] += inc_frac
                        fd["w"] = float(min(1.0, fd["w"] + inc_frac))
            x_t = x_t + extra_phys

        # Now update v after purchases so v ≤ w
        for fd in flex_drivers:
            fd["v"] = float(min(1.0, fd["v"] + fd["prev_z"]))

        storage_state = storage_state + x_t - z_t
        x_prev_global = x_t
        x_list.append(x_t)
        z_list.append(z_t)
        s_list.append(storage_state)

    return x_list, z_list, s_list

# -------------------------
# Main: build analytical thresholds, run PALD/PAAD/OPT once, print ratios
# -------------------------
def main():
    # Load a single instance
    price_all, base_all, flex_all, Delta_all, p_min, p_max = load_scenarios_with_flexible(1, T, trace)
    p0 = price_all[0]
    b0 = base_all[0]
    f0 = flex_all[0]
    Delta0 = Delta_all[0]

    # Analytical alpha and thresholds
    alpha = float(get_alpha(float(p_min), float(p_max), c_delivery, eps_delivery, T, gamma, delta))
    w_grid = [(i + 0.5) / K for i in range(K)]
    y_base = [base_threshold(w, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha, b=1.0) for w in w_grid]
    # enforce non-increasing and end-value pins (to match training conventions)
    for i in range(1, K):
        y_base[i] = min(y_base[i], y_base[i-1])
    if K > 0:
        y_base[-1] = float(p_min) + 2.0 * gamma

    # Use analytical flexible thresholds from PAAD
    y_flex_p = [flex_purchase_threshold(w, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha, f=1.0) for w in w_grid]
    for i in range(1, K):
        y_flex_p[i] = min(y_flex_p[i], y_flex_p[i-1])
    if K > 0:
        y_flex_p[-1] = float(p_min) + 2.0 * gamma

    y_flex_d = [flex_delivery_threshold(v, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha, f=1.0) for v in w_grid]
    for i in range(1, K):
        y_flex_d[i] = min(y_flex_d[i], y_flex_d[i-1])
    if K > 0:
        y_flex_d[-1] = float(p_min) * (c_delivery + eps_delivery) + 2.0 * delta

    # Build manually tuned thresholds (third algorithm)
    y_tuned_base, y_tuned_flex_p, y_tuned_flex_d = build_tuned_thresholds(
        K, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha
    )

    # PALD forward
    # pald_x, pald_z, pald_s = forward_pald(p0, b0, f0, Delta0, y_base, y_flex_p, y_flex_d, K, gamma, delta, c_delivery, eps_delivery)
    # pald_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, pald_x, pald_z)
    # PALD fast (CVXPY + Clarabel)
    pald_fast_x, pald_fast_z, pald_fast_s = forward_pald_fast(p0, b0, f0, Delta0, y_base, y_flex_p, y_flex_d, K, gamma, delta, c_delivery, eps_delivery)
    pald_fast_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, pald_fast_x, pald_fast_z)
    # PALD with tuned thresholds
    # pald_tuned_x, pald_tuned_z, pald_tuned_s = forward_pald(p0, b0, f0, Delta0, y_tuned_base, y_tuned_flex_p, y_tuned_flex_d, K, gamma, delta, c_delivery, eps_delivery)
    # pald_tuned_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, pald_tuned_x, pald_tuned_z)

    # PAAD baseline
    paad_res = pi.paad_algorithm(T, p0, gamma, delta, c_delivery, eps_delivery, p_min, p_max, S, b0, f0, Delta0)
    paad_x = paad_res['x']
    paad_z = paad_res['z']
    paad_s = paad_res['s'][1:]
    paad_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, paad_x, paad_z)

    # Offline OPT (if available)
    opt_cost = None
    opt_delivered = None
    if _HAS_GUROBI and opt_sol is not None:
        try:
            status, results = opt_sol.optimal_solution(T, p0, gamma, delta, c_delivery, eps_delivery, S, b0, f0, Delta0)
            if status == "Optimal" and results is not None:
                opt_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, results['x'], results['z'])
                opt_delivered = float(sum(results['z']))
        except Exception as e:
            print(f"OPT failed: {e}")

    # Print results
    print("Sanity check with analytical thresholds:")
    print(f"PAAD objective: {paad_cost:.4f}")
    # print(f"PALD objective: {pald_cost:.4f}")
    print(f"PALD-Fast objective: {pald_fast_cost:.4f}")
    # print(f"PALD-Tuned objective: {pald_tuned_cost:.4f}")
    if opt_cost is not None and opt_cost > 0:
        print(f"OPT objective:  {opt_cost:.4f}")
        print(f"PAAD/OPT ratio: {paad_cost/opt_cost:.4f}")
        # print(f"PALD/OPT ratio: {pald_cost/opt_cost:.4f}")
        print(f"PALD-Fast/OPT ratio: {pald_fast_cost/opt_cost:.4f}")
        # print(f"PALD-Tuned/OPT ratio: {pald_tuned_cost/opt_cost:.4f}")

    # Demand and delivered totals
    total_demand = float(sum(b0) + sum(f0))
    # pald_delivered = float(sum(pald_z))
    pald_fast_delivered = float(sum(pald_fast_z))
    # pald_tuned_delivered = float(sum(pald_tuned_z))
    paad_delivered = float(sum(paad_z))
    print("\nDelivered totals")
    print(f"Total demand (base + flex): {total_demand:.4f}")
    print(f"Delivered by PAAD:          {paad_delivered:.4f}")
    # print(f"Delivered by PALD:          {pald_delivered:.4f}")
    print(f"Delivered by PALD-Fast:     {pald_fast_delivered:.4f}")
    # print(f"Delivered by PALD-Tuned:    {pald_tuned_delivered:.4f}")
    if opt_delivered is not None:
        print(f"Delivered by OPT:           {opt_delivered:.4f}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
        t = list(range(T))
        fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        axes[0].plot(t, p0, label='price', color='black')
        axes[0].legend(); axes[0].set_ylabel('price')

        # axes[1].plot(t, pald_x, label='PALD x', color='tab:blue')
        axes[1].plot(t, pald_fast_x, label='PALD-Fast x', color='tab:purple', linestyle='-.')
        # axes[1].plot(t, pald_tuned_x, label='PALD-Tuned x', color='tab:green', linestyle='--')
        axes[1].plot(t, paad_x, label='PAAD x', color='tab:orange')
        axes[1].legend(); axes[1].set_ylabel('x')

        # axes[2].plot(t, pald_z, label='PALD z', color='tab:blue')
        axes[2].plot(t, pald_fast_z, label='PALD-Fast z', color='tab:purple', linestyle='-.')
        # axes[2].plot(t, pald_tuned_z, label='PALD-Tuned z', color='tab:green', linestyle='--')
        axes[2].plot(t, paad_z, label='PAAD z', color='tab:orange')
        axes[2].legend(); axes[2].set_ylabel('z')

        # axes[3].plot(t, pald_s, label='PALD s', color='tab:blue')
        axes[3].plot(t, pald_fast_s, label='PALD-Fast s', color='tab:purple', linestyle='-.')
        # axes[3].plot(t, pald_tuned_s, label='PALD-Tuned s', color='tab:green', linestyle='--')
        axes[3].plot(t, paad_s, label='PAAD s', color='tab:orange')
        axes[3].legend(); axes[3].set_ylabel('s'); axes[3].set_xlabel('t')

        plt.tight_layout()
        plt.savefig('sanity_check_pald_vs_paad.png', dpi=150)
        print("Saved sanity_check_pald_vs_paad.png")
    except Exception as e:
        print(f"Plotting skipped: {e}")

if __name__ == "__main__":
    main()