import torch
import os
import argparse
import torch.optim as optim
from functions import load_scenarios_with_flexible
from pald_implementation import (
    make_pald_base_layer,
    make_pald_flex_purchase_layer,
    make_pald_flex_delivery_layer,
)
from paad_implementation import get_alpha
from robust_projection import project_y_robust, project_y_flex_robust
import paad_implementation as pi
import math
from paad_implementation import objective_function as np_objective_function
import cvxpy as cp
import pickle
import opt_sol
from tqdm import tqdm
from datetime import datetime  # [ADD] run start time + tag

torch.set_num_threads(os.cpu_count() or 4)

# [ADD] Capture run start time/tag and a list of generated files
run_start_dt = datetime.now()
run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")
run_tag = run_start_dt.strftime("%m%d%H%M")
print(f"[run] Start time: {run_start_str} (tag={run_tag})")
run_generated_files: list[str] = []

solver_options = { "solve_method": "ECOS", "abstol": 1e-5, "reltol": 1e-5, "feastol": 1e-5 }

parser = argparse.ArgumentParser(description="Train PALD with flexible demand and deadlines.")
parser.add_argument('--batch_size', type=int, default=25, help='Batch size for training (default: 25)')
parser.add_argument('--num_batches', type=int, default=1, help='Number of batches per epoch (default: 1)')
parser.add_argument('--use_cost_loss', action='store_true', help='Use total cost loss instead of competitive-ratio loss')
parser.add_argument('--pretraining_loop', type=int, default=100, help='Number of epochs to pretrain with random thresholds (default: 100)')
parser.add_argument('--trace', type=str, default="CAISO", help='Trace name to use (default: CAISO)')
parser.add_argument('--month', type=int, default=1, help='Month to filter for in trace (default: 1)')
args = parser.parse_args()

K = 10           # number of segments in piecewise linear approximation for psi
gamma = 10.0     # switching cost parameter for x
delta = 5.0     # switching cost parameter for z (used in analytical threshold)
S = 1.0          # maximum inventory capacity
T = 48          # 12 hours in 15-minute intervals
c_delivery = 0.2
eps_delivery = 0.05
epochs = 200
# get batch size from command line 
batch_size = args.batch_size
# get length of pretraining loop from command line
pretraining_loop = args.pretraining_loop
# get number of batches from command line
num_batches = args.num_batches
# use total cost loss flag
use_cost_loss = args.use_cost_loss
# get trace name from command line
trace = args.trace
month = args.month
learning_rate = 1.0

# Prefetch all scenarios for all batches (e.g., 25 * 100 = 2500)
total_instances = batch_size * num_batches
price_all, base_all, flex_all, Delta_all, p_min, p_max = load_scenarios_with_flexible(total_instances, T, trace, month=month)

# compute the minimum and maximum prices across the actual instances we consider
all_prices_flat = [price for seq in price_all for price in seq]
price_min = min(all_prices_flat) if all_prices_flat else float('inf')
price_max = max(all_prices_flat) if all_prices_flat else float('-inf')

alpha = float(get_alpha(float(p_min), float(p_max), c_delivery, eps_delivery, 96, gamma, delta))
print(f"Computed alpha for analytical thresholds: {alpha}")
beta = 5*alpha

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

    if os.path.exists(f"opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl"):
        with open(f"opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl", "rb") as f:
            opt_costs, total_demands_saved = pickle.load(f)
        print(f"Loaded precomputed OPT costs for flexible demand from opt_sols/opt_costs_flex_{trace}_{month}_{total_instances}.pkl")

        # verify that the saved total demands match
        if total_demands != total_demands_saved:
            print("Warning: Total demands do not match the saved values, recomputing OPT costs.") # force recomputation
            opt_costs = []  # force recomputation
        else:
            return opt_costs, total_demands

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

# Keep analytical threshold utilities for plotting comparison
def base_threshold(w: float, p_min: float, p_max: float, gamma: float, delta: float, c: float, eps: float, T: int, alpha: float, b: float = 1.0) -> float:
    lhs = p_max + 2.0 * gamma + p_min * c
    inside_exp = (p_max * (1.0 + c + eps) + 2.0 * (gamma + delta)) / alpha - (p_max * (1.0 + eps) + p_min * c + 2.0 * (gamma + delta) / T)
    return lhs + inside_exp * math.exp(w / (alpha * max(b, 1e-8)))
def flex_purchase_threshold(w: float, p_min: float, p_max: float, gamma: float, delta: float, c: float, eps: float, T: int, alpha: float, f: float = 1.0) -> float:
    alpha_p = alpha * (1.0 + eps) / (1.0 + c + eps)
    omega = (1.0 + c + eps) / (1.0 + eps)
    lhs = p_max + 2.0 * gamma + p_min * c
    inside = (p_max + 2.0 * gamma) / alpha_p - (p_max + p_min * c + (2.0 * gamma / T) * omega)
    return lhs + inside * math.exp(w / (alpha_p * max(f, 1e-8)))
def flex_delivery_threshold(v: float, p_min: float, p_max: float, gamma: float, delta: float, c: float, eps: float, T: int, alpha: float, f: float = 1.0) -> float:
    alpha_p = alpha * (1.0 + eps) / (1.0 + c + eps)
    omega = (1.0 + c + eps) / (1.0 + eps)
    lhs = p_max * (c + eps) + 2.0 * delta
    inside = (p_max * (c + eps) + 2.0 * delta) / alpha_p - (p_max * (c + eps) + (2.0 * delta / T) * omega)
    return lhs + inside * math.exp(v / (alpha_p * max(f, 1e-8)))

# Midpoint grid per segment
w_grid = [(i + 0.5) / K for i in range(K)]

# Base threshold init (monotone, tail pin)
y_init_base = [base_threshold(w, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha, b=1.0) for w in w_grid]
for i in range(1, K):
    y_init_base[i] = min(y_init_base[i], y_init_base[i-1])
if K > 0:
    y_init_base[-1] = float(p_min) + 2.0 * gamma

# Flexible purchase threshold init
y_init_flex_p = [flex_purchase_threshold(w, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha, f=1.0) for w in w_grid]
for i in range(1, K):
    y_init_flex_p[i] = min(y_init_flex_p[i], y_init_flex_p[i-1])
if K > 0:
    y_init_flex_p[-1] = float(p_min) + 2.0 * gamma

# Flexible delivery threshold init
y_init_flex_d = [flex_delivery_threshold(v, float(p_min), float(p_max), gamma, delta, c_delivery, eps_delivery, T, alpha, f=1.0) for v in w_grid]
for i in range(1, K):
    y_init_flex_d[i] = min(y_init_flex_d[i], y_init_flex_d[i-1])
if K > 0:
    y_init_flex_d[-1] = float(p_min) * (c_delivery + eps_delivery) + 2.0 * delta

# Create learnable tensors from analytical init
y = torch.nn.Parameter(torch.tensor(y_init_base, dtype=torch.float32))
y_flex_p = torch.nn.Parameter(torch.tensor(y_init_flex_p, dtype=torch.float32))  # flexible purchase threshold
y_flex_d = torch.nn.Parameter(torch.tensor(y_init_flex_d, dtype=torch.float32))  # flexible delivery threshold


pald_base_layer = make_pald_base_layer(K, gamma)
pald_flex_purchase_layer = make_pald_flex_purchase_layer(K, gamma)
pald_flex_delivery_layer = make_pald_flex_delivery_layer(K, delta, c_delivery, eps_delivery)
# optimizer = optim.Adam([y, y_flex_p, y_flex_d], lr=learning_rate)
# use SGD instead
optimizer = optim.SGD([y, y_flex_p, y_flex_d], lr=learning_rate)

# Track the best thresholds seen during training (by total loss)
best_snapshot = {
    "loss": float("inf"),
    "epoch": -1,
    "y": None,
    "yp": None,
    "yd": None,
}

# if a saved model exists, load it
# if os.path.exists("best_thresholds_{}_{}.pkl".format(trace, batch_size)):
#     with open("best_thresholds_{}_{}.pkl".format(trace, batch_size), "rb") as f:
#         best_snapshot = pickle.load(f)
#         print(f"Loaded saved best thresholds with loss {best_snapshot['loss']:.4f}.")

def compute_segment_caps(w_prev: float, K: int):
    """Remaining capacity per segment given cumulative fraction w_prev."""
    # Clamp w into [0, 1]
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

def torch_objective(p_seq, x_seq, z_seq, gamma, delta, c, eps):
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
    switching_cost_x = gamma * (x_seq[1:] - x_seq[:-1]).abs().sum() if Tn > 1 else torch.tensor(0.0)
    switching_cost_z = delta * (z_seq[1:] - z_seq[:-1]).abs().sum() if Tn > 1 else torch.tensor(0.0)
    # Note: s_{t-1} term via roll(1); s_{-1} uses previous s_T but is unused because multiplied with z[0]; fix first index:
    s_prev_seq = torch.cat([s_torch[:-1]])
    discharge_cost = (p_seq * (c * z_seq + eps * z_seq - c * s_prev_seq * z_seq)).sum()
    return cost_purchasing + switching_cost_x + switching_cost_z + discharge_cost

def precompute_opt_costs(price_instances, demand_instances, T, gamma, delta, c, eps, S):
    opt_costs = []
    for p_seq, b_seq in zip(price_instances, demand_instances):
        f = [0.0 for _ in range(T)]
        Delta_f = [0 for _ in range(T)]
        try:
            status, results = opt_sol.optimal_solution(T, p_seq, gamma, delta, c, eps, S, b_seq, f, Delta_f)
            if status == "Optimal" and results is not None:
                # Compute objective via numpy function to ensure consistency
                opt_cost = np_objective_function(T, p_seq, gamma, delta, c, eps, results['x'], results['z'])
            else:
                opt_cost = None
        except Exception:
            opt_cost = None
        opt_costs.append(opt_cost)
    return opt_costs

def _safe_layer_call(layer, args, size=1.0):
    """
    Call a CvxpyLayer and catch SCS/diffcp failures. Returns a scalar tensor.
    """
    # try:
    (val,) = layer(*args, solver_args=solver_options)  # solver_args already bound on layer
    return torch.clamp(val, min=0.0, max=size)



# Main training loop
losses = []
reinitialized = 0
try: 
    for epoch in range(epochs):
        # if we are still in the pretraining loop, we randomly reinitialize y each epoch
        if epoch < pretraining_loop and epoch > 0:
            # reinitialize y, y_flex_p, y_flex_d to random values between p_min and p_max
            with torch.no_grad():
                price_batch = price_all[start:end]
                price_seq = price_batch[0]
                y_random = torch.rand(K) * (float(price_max) - float(price_min)) + float(price_min)
                y_random, _ = torch.sort(y_random, descending=True)
                y_random[-1] = float(p_min) + 2.0 * float(gamma)
                y.copy_(y_random)

                y_flex_p_random = torch.rand(K) * (float(price_max) - float(price_min)) + float(price_min)
                y_flex_p_random, _ = torch.sort(y_flex_p_random, descending=True)
                y_flex_p_random[-1] = float(price_min) + 2.0 * float(gamma)
                y_flex_p.copy_(y_flex_p_random)

                y_flex_d_random = torch.rand(K) * (float(price_max) * (c_delivery + eps_delivery) - float(price_min) * (c_delivery + eps_delivery)) + float(price_min) * (c_delivery + eps_delivery)
                y_flex_d_random, _ = torch.sort(y_flex_d_random, descending=True)
                y_flex_d_random[-1] = float(price_min) * (c_delivery + eps_delivery) + 2.0 * float(delta)
                y_flex_d.copy_(y_flex_d_random)

                # project to ensure robustness
                y_proj = project_y_robust(y, K, float(p_min), float(p_max), float(gamma), float(delta), float(c_delivery), float(eps_delivery), int(T), beta=beta)
                if y_proj is not None:
                    for i in range(K):
                        y[i] = torch.tensor(float(y_proj[i]))
                if K > 0:
                    y[-1] = torch.tensor(float(p_min) + 2.0 * gamma)
                phi_proj, psi_proj = project_y_flex_robust(
                    y_flex_p, y_flex_d, K,
                    float(p_min), float(p_max),
                    float(gamma), float(delta),
                    float(c_delivery), float(eps_delivery),
                    int(T), beta=beta
                )
                for i in range(K):
                    y_flex_p[i] = torch.tensor(float(phi_proj[i]))
                    y_flex_d[i] = torch.tensor(float(psi_proj[i]))
                # clamp to max ranges (safety)
                for i in range(K):
                    y[i] = torch.clamp(y[i], max=float(p_max))
                    y_flex_p[i] = torch.clamp(y_flex_p[i], max=float(p_max))
                    y_flex_d[i] = torch.clamp(y_flex_d[i], max=float(p_max) * (c_delivery + eps_delivery))

            # clear the losses
            losses = []
        
        if epoch == pretraining_loop:
            print(f"Pretraining completed after {pretraining_loop} epochs")
            # load the best snapshot so far
            if best_snapshot["y"] is not None:
                with torch.no_grad():
                    y.copy_(best_snapshot["y"])
                    y_flex_p.copy_(best_snapshot["yp"])
                    y_flex_d.copy_(best_snapshot["yd"])
                print(f"Loaded best snapshot from pretraining with loss {best_snapshot['loss']:.4f} at epoch {best_snapshot['epoch']}.")

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            price_batch = price_all[start:end]
            base_batch = base_all[start:end]
            flex_batch = flex_all[start:end]
            Delta_batch = Delta_all[start:end]
            batch_total_loss = torch.tensor(0.0)
            # Iterate instances in this batch
            for idx, (price_seq, base_seq, flex_seq, Delta_seq) in enumerate(zip(price_batch, base_batch, flex_batch, Delta_batch)):
                global_idx = start + idx  # align with precomputed OPT list
                # global storage state in physical units
                storage_state = 0.0
                x_prev_global = torch.tensor(0.0)

                # Each base driver tracks fractional progress (unit capacity); demand scales the fractional decision
                base_drivers = []  # list of dicts with keys: id, b (demand), w (fraction), prev_decision (fraction)
                base_drivers.append({"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0})
                # Flexible drivers: track purchase (w) and delivery (v) progress fractions
                flex_drivers = []  # dict keys: id, f, delta, w, v, prev_x, prev_z

                inst_cost = torch.tensor(0.0)
                x_hist = []
                z_hist = []

                for t in range(T):
                    b_t_val = float(base_seq[t])
                    p_t_val = float(price_seq[t])
                    # Add flexible driver arrivals
                    f_arrival = float(flex_seq[t])
                    dlt = int(Delta_seq[t])
                    if f_arrival > 0:
                        flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": dlt, "w": 0.0, "v": 0.0, "prev_x": 0.0, "prev_z": 0.0})

                    # Add a new driver if there is base demand arrival
                    if b_t_val > 0:
                        base_drivers.append({"id": 2 * t + 2, "b": b_t_val, "w": 0.0, "prev_decision": 0.0})

                    # Compute purchasing excess from previous step in physical units
                    prev_purchasing_total = 0.0
                    for drv in base_drivers:
                        prev_purchasing_total += drv["prev_decision"] * drv["b"]
                    for fd in flex_drivers:
                        prev_purchasing_total += fd["prev_x"] * fd["f"]
                    purchasing_excess = x_prev_global.item() - prev_purchasing_total

                    # Compute delivery excess from previous step in physical units
                    prev_delivery_total = 0.0
                    for fd in flex_drivers:
                        prev_delivery_total += fd["prev_z"] * fd["f"]
                    # last z was base b_{t-1} + flex deliveries; but we only need per-driver shares here

                    # Determine per-driver pseudo decisions (fractional)
                    decisions = []  # list of tensors in physical units
                    for drv in base_drivers:
                        b_i = drv["b"]
                        prev_frac = drv["prev_decision"]
                        # share positive excess proportional to previous physical contribution
                        denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
                        share = (prev_frac * b_i) / denom if prev_purchasing_total > 0 else 0.0
                        pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(b_i, 1e-8)

                        # Compute per-segment caps for current cumulative fraction w
                        w_prev_frac = float(drv["w"])
                        # Clamp w into [0, 1 - eps]
                        w_eff = max(0.0, min(1.0 - 1e-9, w_prev_frac))
                        caps_list = compute_segment_caps(w_eff, K)
                        if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                            cur_frac_decision = torch.tensor(0.0, dtype=torch.float32)
                            cur_phys_decision = torch.mul(cur_frac_decision, b_i)
                            decisions.append(cur_phys_decision)
                            drv["prev_decision"] = float(cur_frac_decision.detach())
                            drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))
                            continue
                        # Call CVX layer with full y vector and caps (all tensors)
                        x_prev_frac_t = torch.tensor(float(pseudo_prev_frac), dtype=torch.float32)
                        w_prev_frac_t = torch.tensor(w_eff, dtype=torch.float32)
                        p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                        y_vec_t = y
                        caps_t = torch.tensor(caps_list, dtype=torch.float32)

                        cur_frac_decision = _safe_layer_call(
                            pald_base_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y_vec_t, caps_t), size=(1.0 - w_eff)
                        )

                        # Convert to physical units by scaling with demand of this driver
                        cur_phys_decision = torch.mul(cur_frac_decision, b_i)
                        decisions.append(cur_phys_decision)

                        # Update driver internal state (detach to avoid history growth)
                        drv["prev_decision"] = float(cur_frac_decision.detach())
                        drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))

                    # Flexible drivers: purchasing decisions
                    for fd in flex_drivers:
                        f_i = fd["f"]
                        prev_frac_x = fd["prev_x"]
                        denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
                        share = (prev_frac_x * f_i) / denom if prev_purchasing_total > 0 else 0.0
                        pseudo_prev_x = prev_frac_x + max(0.0, purchasing_excess) * share / max(f_i, 1e-8)

                        w_prev_frac = float(fd["w"])
                        w_eff = max(0.0, min(1.0 - 1e-9, w_prev_frac))

                        # Enforce deadline and purchase cap outside the layer (keeps DPP)
                        if t >= max(0, int(fd["delta"]) - 1):
                            cur_frac_x = torch.tensor(max(0.0, 1.0 - w_prev_frac), dtype=torch.float32)
                        else:
                            caps_list = compute_segment_caps(w_eff, K)
                            if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                                cur_frac_x = torch.tensor(0.0, dtype=torch.float32)
                                cur_phys_x = torch.mul(cur_frac_x, f_i)
                                decisions.append(cur_phys_x)
                                fd["prev_x"] = float(cur_frac_x.detach())
                                fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))
                                continue
                            x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_x)))
                            x_prev_frac_t = torch.tensor(x_prev_clamped, dtype=torch.float32)
                            w_prev_frac_t = torch.tensor(w_eff, dtype=torch.float32)
                            p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                            y_vec_t = y_flex_p
                            caps_t = torch.tensor(caps_list, dtype=torch.float32)

                            cur_frac_x = _safe_layer_call(
                                pald_flex_purchase_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y_vec_t, caps_t), size=(1.0 - w_eff)
                            )
                        cur_phys_x = torch.mul(cur_frac_x, f_i)
                        decisions.append(cur_phys_x)
                        fd["prev_x"] = float(cur_frac_x.detach())
                        fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))

                    # Aggregate physical purchases this step
                    x_t = torch.stack(decisions).sum() if decisions else torch.tensor(0.0)

                    # Base delivery equals current base demand arrival
                    z_components = [torch.tensor(b_t_val, dtype=torch.float32)]

                    # Flexible drivers: delivery decisions
                    for fd in flex_drivers:
                        f_i = fd["f"]
                        prev_frac_z = fd["prev_z"]
                        # share delivery excess (if you track it globally); here we just use prev_frac_z
                        v_prev_frac = float(fd["v"])
                        w_prev_frac = float(fd["w"])

                        # Enforce deadline and purchase cap outside the layer (keeps DPP)
                        if t >= max(0, int(fd["delta"]) - 1):
                            cur_frac_z = torch.tensor(max(0.0, 1.0 - v_prev_frac), dtype=torch.float32)
                        else:
                            caps_list = compute_segment_caps(v_prev_frac, K)
                            v_eff = max(0.0, min(1.0 - 1e-9, v_prev_frac))
                            caps_list = compute_segment_caps(v_eff, K)
                            if (1.0 - v_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                                cur_frac_z = torch.tensor(0.0, dtype=torch.float32)
                                cur_frac_z = torch.clamp(cur_frac_z, max=max(0.0, w_prev_frac - v_prev_frac))
                                cur_phys_z = torch.mul(cur_frac_z, f_i)
                                z_components.append(cur_phys_z)
                                fd["prev_z"] = float(cur_frac_z.detach())
                                fd["v"] = float(min(1.0, fd["v"] + fd["prev_z"]))
                                continue
                            z_prev_clamped = max(0.0, min(1.0 - v_eff, float(fd["prev_z"])))
                            z_prev_frac_t = torch.tensor(z_prev_clamped, dtype=torch.float32)
                            v_prev_frac_t = torch.tensor(v_eff, dtype=torch.float32)
                            p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                            s_prev_t = torch.tensor(float(max(0.0, storage_state)), dtype=torch.float32)
                            y_vec_t = y_flex_d
                            caps_t = torch.tensor(caps_list, dtype=torch.float32)

                            # Precompute coeff = p_t * (c+eps) - p_t * c * s_prev  (scalar)
                            coeff_t = torch.tensor(
                                p_t_val * ((c_delivery + eps_delivery) - c_delivery * float(max(0.0, storage_state))),
                                dtype=torch.float32,
                            )

                            cur_frac_z = _safe_layer_call(
                                pald_flex_delivery_layer, (z_prev_frac_t, v_prev_frac_t, coeff_t, y_vec_t, caps_t), size=(1.0 - v_eff)
                            )
                        cur_phys_z = torch.mul(cur_frac_z, f_i)
                        z_components.append(cur_phys_z)
                        fd["prev_z"] = float(cur_frac_z.detach())
                        fd["v"] = float(min(1.0, fd["v"] + fd["prev_z"]))

                    z_t = torch.stack(z_components).sum()

                    # Ensure purchases cover deliveries (inventory feasibility)
                    x_t = torch.maximum(x_t, z_t - torch.tensor(storage_state, dtype=torch.float32))
                    # Ensure we do not exceed max storage capacity
                    x_t = torch.clamp(x_t, max=torch.tensor(z_t + S - storage_state, dtype=torch.float32))
                    # Ensure nonnegative
                    x_t = torch.clamp(x_t, min=torch.tensor(0.0, dtype=torch.float32))
                    storage_state = float(storage_state + float(x_t.detach()) - float(z_t.detach()))

                    inst_cost = inst_cost + p_t_t * x_t + gamma * torch.abs(x_t - x_prev_global)
                    x_prev_global = x_t.detach()

                    # record sequences for torch objective
                    x_hist.append(x_t)
                    z_hist.append(z_t)

                # Convert sequences for torch objective
                p_torch = torch.tensor([float(v) for v in price_seq], dtype=torch.float32)
                x_torch = torch.stack(x_hist) if x_hist else torch.ones(T)
                z_torch = torch.stack(z_hist) if z_hist else torch.zeros(T)

                pald_cost = torch_objective(p_torch, x_torch, z_torch, gamma, delta, c_delivery, eps_delivery)

                if not use_cost_loss:
                    # # Competitive-ratio loss: ReLU(pald/opt - 1), opt treated as constant
                    inst_loss = torch.tensor(0.0)
                    if opt_costs_all is not None:
                        opt_val = opt_costs_all[global_idx]
                        if opt_val is not None and opt_val > 1e-6:
                            # inst_loss = torch.relu(pald_cost - torch.tensor(float(opt_val), dtype=torch.float32))

                            # Alternative: relative gap scaled up
                            denom = torch.tensor(float(opt_val), dtype=torch.float32)
                            inst_loss = torch.relu(((pald_cost / denom) - 1.0)*1000.0)  # scale up to keep similar magnitude
                    batch_total_loss = batch_total_loss + inst_loss

                    # # check that the total delivery of pald is close to the total demand
                    # print("total delivered: ", torch.sum(z_torch).item(), " total demand: ", sum(base_seq) + sum(flex_seq))
                    # print(" pald cost: ", pald_cost.item(), " opt cost: ", None if opt_costs_all is None else opt_costs_all[global_idx], " inst loss: ", inst_loss.item())

                    # # check that the torch_objective matches the np_objective_function
                    # np_cost = np_objective_function(T, [float(v) for v in price_seq], gamma, delta, c_delivery, eps_delivery, [float(v.detach()) for v in x_torch], [float(v.detach()) for v in z_torch])
                    # print(" np cost: ", np_cost, " pald cost: ", pald_cost.item())

                else:
                    # Total cost loss
                    batch_total_loss = batch_total_loss + pald_cost
                

            # end for each instance

            # Normalize by batch size
            batch_total_loss = batch_total_loss / batch_size

            # Regularize y to be non-increasing: sum(ReLU(y[i+1] - y[i])) for all three thresholds
            mono_penalty = torch.relu(y[1:] - y[:-1]).sum()
            mono_penalty = mono_penalty + 0.05 * torch.relu(y_flex_p[1:] - y_flex_p[:-1]).sum()
            mono_penalty = mono_penalty + 0.05 * torch.relu(y_flex_d[1:] - y_flex_d[:-1]).sum()

            # Final loss: CR loss + small monotonicity penalty
            loss = batch_total_loss + 0.1 * mono_penalty
            losses.append(float(loss.item()))

            # if we are not in the pretraining loop, we do normal optimization
            if epoch >= pretraining_loop:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Project y onto robustness sets every 10 epochs
                with torch.no_grad():
                    y_proj = project_y_robust(y, K, float(p_min), float(p_max), float(gamma), float(delta), float(c_delivery), float(eps_delivery), int(T), beta=beta)
                    if y_proj is not None:
                        for i in range(K):
                            y[i] = torch.tensor(float(y_proj[i]))
                    if K > 0:
                        y[-1] = torch.tensor(float(p_min) + 2.0 * gamma)
                    phi_proj, psi_proj = project_y_flex_robust(
                        y_flex_p, y_flex_d, K,
                        float(p_min), float(p_max),
                        float(gamma), float(delta),
                        float(c_delivery), float(eps_delivery),
                        int(T), beta=beta
                    )
                    for i in range(K):
                        y_flex_p[i] = torch.tensor(float(phi_proj[i]))
                        y_flex_d[i] = torch.tensor(float(psi_proj[i]))
                    # clamp to max ranges (safety)
                    for i in range(K):
                        y[i] = torch.clamp(y[i], max=float(p_max))
                        y_flex_p[i] = torch.clamp(y_flex_p[i], max=float(p_max))
                        y_flex_d[i] = torch.clamp(y_flex_d[i], max=float(p_max) * (c_delivery + eps_delivery))

            if (batch_idx % 10 == 0) and (epoch % 10 == 0):
                # Optionally log average CR over this batch (where OPT exists)
                with torch.no_grad():
                    cr_vals = []
                    for idx, (price_seq, _, _, _) in enumerate(zip(price_batch, base_all[start:end], flex_batch, Delta_batch)):
                        global_idx = start + idx
                        opt_val = None if opt_costs_all is None else opt_costs_all[global_idx]
                        if opt_val is not None and opt_val > 1e-6:
                            # recompute pald_cost from last forward for logging if desired
                            pass
                    # keep simple to avoid recomputation noise
                print(f"epoch {epoch} batch {batch_idx}, loss {loss.item():.4f}")

            # After projection, update best snapshot if improved
            with torch.no_grad():
                curr = float(loss.item())
                if curr + 1e-12 < best_snapshot["loss"]:
                    best_snapshot["loss"] = curr
                    best_snapshot["epoch"] = epoch
                    best_snapshot["batch"] = batch_idx
                    best_snapshot["y"] = y.detach().cpu().clone()
                    best_snapshot["yp"] = y_flex_p.detach().cpu().clone()
                    best_snapshot["yd"] = y_flex_d.detach().cpu().clone()
                    print(f"[best] Updated at epoch {epoch} batch {batch_idx}: loss={curr:.6f}")
            
            # End-of-epoch diagnostics
            # with torch.no_grad():
            #     # gradient norms (sanity: should be finite and not always ~0)
            #     g_base = y.grad.detach().data.norm().item() if y.grad is not None else float('nan')
            #     g_fp = y_flex_p.grad.detach().data.norm().item() if y_flex_p.grad is not None else float('nan')
            #     g_fd = y_flex_d.grad.detach().data.norm().item() if y_flex_d.grad is not None else float('nan')
            #     def pct(a, b): return (100.0 * a / b) if b > 0 else 0.0
            #     print(f"[epoch {epoch}] grad ||y||={g_base:.3e}, ||yp||={g_fp:.3e}, ||yd||={g_fd:.3e} | ")

except KeyboardInterrupt:
    print("[train] Caught KeyboardInterrupt. Skipping remaining training and running evaluation...")

# Restore best snapshot before saving/plotting/evaluation
with torch.no_grad():
    if best_snapshot["y"] is not None:
        print(f"[best] Restoring best thresholds from epoch {best_snapshot['epoch']} batch {best_snapshot['batch']} (loss={best_snapshot['loss']:.6f})")
        y.data = best_snapshot["y"].to(y.device)
        y_flex_p.data = best_snapshot["yp"].to(y_flex_p.device)
        y_flex_d.data = best_snapshot["yd"].to(y_flex_d.device)

        # [CHG] Save tagged best-thresholds file
        os.makedirs(".", exist_ok=True)
        import pickle
        best_outfile = f"best_thresholds_{trace}_{month}_{batch_size}_{run_tag}.pkl"
        with open(best_outfile, 'wb') as f:
            pickle.dump({
                'y_base': y.detach().cpu().numpy().tolist(),
                'y_flex_purchase': y_flex_p.detach().cpu().numpy().tolist(),
                'y_flex_delivery': y_flex_d.detach().cpu().numpy().tolist(),
                'loss': best_snapshot["loss"],
                'epoch': best_snapshot["epoch"],
                'batch': best_snapshot["batch"],
                'run_start': run_start_str,
                'run_tag': run_tag,
            }, f)
        run_generated_files.append(best_outfile)
        print(f"Saved {best_outfile}")

        # [ADD] Print the best-found parameter vectors
        def _fmt_vec(t):
            return "[" + ", ".join(f"{float(v):.6f}" for v in t.detach().cpu().flatten().tolist()) + "]"
        print("Best y_base:", _fmt_vec(y))
        print("Best y_flex_purchase:", _fmt_vec(y_flex_p))
        print("Best y_flex_delivery:", _fmt_vec(y_flex_d))


# Evaluate PALD, PAAD, and OPT on the first instance and plot time series
# -------------------------
# Forward PALD with given thresholds
# -------------------------
def forward_pald(price_seq, base_seq, flex_seq, Delta_seq):
    x_list, z_list, s_list = [], [], []
    storage_state = 0.0
    x_prev_global = torch.tensor(0.0)

    # thresholds as tensors
    yp = y_flex_p
    yd = y_flex_d

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
                        pald_base_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y, caps_t), size=(1.0 - w_eff)
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
                        pald_flex_purchase_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, yp, caps_t), size=(1.0 - w_eff)
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
                        pald_flex_delivery_layer, (z_prev_frac_t, v_prev_frac_t, coeff_t, yd, caps_t), size=(1.0 - v_eff)
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

def evaluate_and_plot_instance0(prefix: str = 'eval_instance0'):
    import matplotlib.pyplot as plt

    if not price_all or not base_all:
        print("No instances available for evaluation.")
        return

    p0 = price_all[0]
    b0 = base_all[0]
    f0 = flex_all[0]
    Delta0 = Delta_all[0]

    # PALD forward pass with learned y
    pald_x, pald_z, pald_s = forward_pald(p0, b0, f0, Delta0)

    # PAAD baseline
    paad_res = pi.paad_algorithm(T, p0, gamma, delta, c_delivery, eps_delivery, p_min, p_max, S, b0, f0, Delta0)
    paad_x = paad_res['x']
    paad_z = paad_res['z']
    paad_s = paad_res['s'][1:]  # drop initial

    # Offline OPT (if available)
    opt_x = opt_z = opt_s = None
    opt_cost = None
    try:
        status, results = opt_sol.optimal_solution(T, p0, gamma, delta, c_delivery, eps_delivery, S, b0, f0, Delta0)
        if status == "Optimal" and results is not None:
            opt_x = results['x']
            opt_z = results['z']
            opt_s = results['s'][1:]
            # Use numpy objective for consistency
            opt_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, opt_x, opt_z)
    except Exception as e:
        print(f"OPT evaluation failed: {e}")

    # Time axis
    t = list(range(T))

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(t, p0, label='price', color='black')
    axes[0].set_ylabel('price')
    axes[0].legend()

    axes[1].plot(t, pald_x, label='PALD x', color='tab:blue')
    axes[1].plot(t, paad_x, label='PAAD x', color='tab:orange')
    if opt_x is not None:
        axes[1].plot(t, opt_x, label='OPT x', color='tab:green')
    axes[1].set_ylabel('purchasing x')
    axes[1].legend()

    axes[2].plot(t, pald_z, label='PALD z', color='tab:blue')
    axes[2].plot(t, paad_z, label='PAAD z', color='tab:orange')
    if opt_z is not None:
        axes[2].plot(t, opt_z, label='OPT z', color='tab:green')
    axes[2].set_ylabel('delivery z')
    axes[2].legend()

    axes[3].plot(t, pald_s, label='PALD s', color='tab:blue')
    axes[3].plot(t, paad_s, label='PAAD s', color='tab:orange')
    if opt_s is not None:
        axes[3].plot(t, opt_s, label='OPT s', color='tab:green')
    axes[3].set_ylabel('storage s')
    axes[3].set_xlabel('time t')
    axes[3].legend()

    plt.tight_layout()
    outfile = f"{prefix}.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved {outfile}")

    # Compute and print competitive ratios if OPT cost available
    pald_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, pald_x, pald_z)
    paad_cost = np_objective_function(T, p0, gamma, delta, c_delivery, eps_delivery, paad_x, paad_z)
    if opt_cost is not None and opt_cost > 0:
        print(f"OPT objective: {opt_cost:.4f}")
        print(f"PAAD objective: {paad_cost:.4f}  | Competitive ratio (PAAD/OPT): {paad_cost/opt_cost:.4f}")
        print(f"PALD objective: {pald_cost:.4f}  | Competitive ratio (PALD/OPT): {pald_cost/opt_cost:.4f}")

        # report the total delivered amounts for OPT, PAAD, PALD
        total_opt_z = sum(opt_z) if opt_z is not None else 0.0
        total_paad_z = sum(paad_z) if paad_z is not None else 0.0
        total_pald_z = sum(pald_z) if pald_z is not None else 0.0
        print(f"Total delivered: OPT={total_opt_z:.2f}, PAAD={total_paad_z:.2f}, PALD={total_pald_z:.2f}")
    else:
        print(f"PAAD objective: {paad_cost:.4f}")
        print(f"PALD objective: {pald_cost:.4f}")
    return outfile  # [ADD] return saved filename

# Run evaluation before and after training
# Post-training evaluation
eval_outfile = evaluate_and_plot_instance0(prefix=f'eval_instance0_{run_tag}_{trace}_{month}')
if eval_outfile:
    run_generated_files.append(eval_outfile)

# [ADD] Write a text log for this run
try:
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/train_log_{trace}_{month}_{batch_size}_{run_tag}.txt"
    # Parameters to log
    params = {
        "K": K,
        "gamma": gamma,
        "delta": delta,
        "S": S,
        "T": T,
        "c_delivery": c_delivery,
        "eps_delivery": eps_delivery,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "num_batches": num_batches,
        "total_instances": total_instances,
        "trace": trace,
        "learning_rate": learning_rate,
    }
    def _fmt_vec_list(t):
        return "[" + ", ".join(f"{float(v):.6f}" for v in (t.detach().cpu().flatten().tolist())) + "]"
    with open(log_path, "w") as f:
        f.write(f"Run start: {run_start_str}\n")
        f.write(f"Run tag: {run_tag}\n")
        f.write("Parameters:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        # Best snapshot info
        if best_snapshot["y"] is not None:
            f.write("\nBest snapshot:\n")
            f.write(f"  loss: {best_snapshot['loss']:.6f}\n")
            f.write(f"  epoch: {best_snapshot['epoch']}\n")
            f.write(f"  batch: {best_snapshot['batch']}\n")
            f.write(f"  y_base: {_fmt_vec_list(y)}\n")
            f.write(f"  y_flex_purchase: {_fmt_vec_list(y_flex_p)}\n")
            f.write(f"  y_flex_delivery: {_fmt_vec_list(y_flex_d)}\n")
        # Historical losses
        f.write("\nHistorical loss values:\n")
        f.write(", ".join(f"{lv:.6f}" for lv in losses) + "\n")
        # Files generated
        f.write("\nGenerated files:\n")
        for fp in run_generated_files:
            f.write(f"  {fp}\n")
    print(f"Saved log: {log_path}")
except Exception as e:
    print(f"Log write failed: {e}")
# # save the learned thresholds to a pickle file
# import pickle
# with open('learned_thresholds_{}.pkl'.format(trace), 'wb') as f:
#     pickle.dump({
#         'y_base': y.detach().cpu().numpy().tolist(),
#         'y_flex_purchase': y_flex_p.detach().cpu().numpy().tolist(),
#         'y_flex_delivery': y_flex_d.detach().cpu().numpy().tolist(),
#     }, f)
# print("Saved learned thresholds to learned_thresholds.pkl")





