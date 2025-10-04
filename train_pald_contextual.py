import torch
import os
import argparse
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from functions import load_scenarios_with_flexible_context
from pald_implementation import (
    make_pald_base_layer,
    make_pald_flex_purchase_layer,
    make_pald_flex_delivery_layer,
    torch_objective,
    hinge_from_y_torch
)
from contextual_model import ThresholdPredictor
import paad_implementation as pi
import math
from paad_implementation import objective_function as np_objective_function
import opt_sol
from tqdm import tqdm
from datetime import datetime  # run start time + tag
from typing import Dict, List, Tuple

"""
Train contextual PALD thresholds with differentiable convex layers.

Pipeline:
- Load scenarios with base and flexible demand.
- Predict per-driver monotone thresholds via a contextual NN.
- For each time step, solve small convex subproblems (CVXPyLayers) for purchase/delivery.
- Build a differentiable cost; optionally add warm-up anchors and small shape/topup terms.
- Track diagnostics and evaluate vs. PAAD and OPT on a reference instance.

 Per-step data flow (one instance):
     inputs: price[t], base[t], flex[t], Delta[t], storage_state
     1) arrivals: add base driver if base[t]>0; add flex driver(s) if flex[t]>0 with deadline Delta[t]
     2) features: build per-driver features (time, remaining, month, forecast stats, price, storage)
     3) thresholds: model(features) -> y_base, y_flex_purchase, y_flex_delivery (each monotone length-K)
     4) delivery: solve flex delivery CVX layer per flex driver to propose z fractions; enforce deadlines
     5) capacity: compute buy_cap = S - storage + z_t
     6) base purchase: solve base CVX layer per base driver under segment caps and buy_cap
     7) flex purchase: solve flex purchase CVX layer per flex driver under segment caps and buy_cap
     8) feasibility: if x_t < z_t - storage, apply forced top-up and attribute to deadline jobs
     9) update: storage = clip(storage + x_t - z_t, [0, S]); update drivers' cumulative fractions
    10) objective: accumulate torch objective; repeat for t+1
"""

torch.set_num_threads(os.cpu_count() or 4)

# [ADD] Capture run start time/tag and a list of generated files
run_start_dt = datetime.now()
run_start_str = run_start_dt.strftime("%Y-%m-%d %H:%M:%S")
run_tag = run_start_dt.strftime("%m%d%H%M")
print(f"[run] Start time: {run_start_str} (tag={run_tag})")
run_generated_files: list[str] = []

# solver_options = { "solve_method": "ECOS", "abstol": 1e-5, "reltol": 1e-5, "feastol": 1e-5 }
solver_options = {}

parser = argparse.ArgumentParser(description="Train PALD with flexible demand and deadlines.")
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4)')
parser.add_argument('--num_batches', type=int, default=1, help='Number of batches per epoch (default: 1)')
parser.add_argument('--use_cost_loss', action='store_true', help='Use total cost loss instead of competitive-ratio loss')
parser.add_argument('--trace', type=str, default="CAISO", help='Trace name to use (default: CAISO)')
parser.add_argument('--month', type=int, default=99, help='Month to filter for in trace (default: 1, 99 for all)')
parser.add_argument('--warmup_epochs', type=int, default=100, help='Supervised warm-up epochs to align y0 to OPT/quantile targets')
parser.add_argument('--warmup_lambda', type=float, default=1000.0, help='Weight of warm-up y0 loss during warm-up phase')
parser.add_argument('--y0_margin', type=float, default=10.0, help='Margin to add to OPT avg price for base/flex purchase y0 target')
parser.add_argument('--post_warmup_epochs', type=int, default=0, help='Number of epochs to decay anchor after warm-up')
parser.add_argument('--post_warmup_lambda', type=float, default=1.0, help='Initial weight of post-warmup anchor (decays to 0)')
parser.add_argument('--freeze_trunk_epochs', type=int, default=0, help='Freeze trunk for these epochs after warm-up to avoid collapse')
parser.add_argument('--topup_penalty_lambda', type=float, default=20.0, help='Weight for penalizing reliance on forced top-ups')
args = parser.parse_args()

K = 10           # number of segments in piecewise linear approximation for psi
taus_full_t = torch.linspace(0.0, 1.0, 11, dtype=torch.float32)
gamma = 10.0     # switching cost parameter for x
delta = 5.0     # switching cost parameter for z (used in analytical threshold)
S = 1.0          # maximum inventory capacity
T = 48          # 12 hours in 15-minute intervals
c_delivery = 0.2
eps_delivery = 0.05
epochs = 1000
# get batch size from command line 
batch_size = args.batch_size
# get number of batches from command line
num_batches = args.num_batches
# use total cost loss flag (if not, the code uses competitve ratio loss)
use_cost_loss = args.use_cost_loss
# get trace name from command line
trace = args.trace
month = args.month
learning_rate = 0.0001

# Prefetch all scenarios for all batches (e.g., 25 * 100 = 2500)
total_instances = batch_size * num_batches
price_all, times_all, months_all, forecast_all, base_all, flex_all, Delta_all, p_min, p_max = load_scenarios_with_flexible_context(total_instances, T, trace, month=month, saved=False)

# ---------------------------------------
# Precompute OPT costs for competitive-ratio loss
# ---------------------------------------
def precompute_opt_costs_flex(price_instances, base_instances, flex_instances, Delta_instances,
                              T, gamma, delta, c, eps, S):
    """
    Compute OPT objective values and average accepted purchase price per instance.
    Returns (opt_costs, avg_prices) with length = number of instances.
    """
    opt_costs = []
    avg_prices = []

    # use TQDM for progress bar
    for p_seq, b_seq, f_seq, dlt in tqdm(zip(price_instances, base_instances, flex_instances, Delta_instances)):
        avg_p = None
        try:
            status, results = opt_sol.optimal_solution(T, p_seq, gamma, delta, c, eps, S, b_seq, f_seq, dlt)
            if status == "Optimal" and results is not None:
                opt_cost = np_objective_function(T, p_seq, gamma, delta, c, eps, results['x'], results['z'])
                x_opt = results['x']
                denom = sum(x_opt)
                if denom and denom > 1e-9:
                    num = sum(pt * xt for pt, xt in zip(p_seq, x_opt))
                    avg_p = float(num / denom)
                else:
                    avg_p = None
            else:
                opt_cost = None
        except Exception:
            opt_cost = None
        opt_costs.append(opt_cost)
        avg_prices.append(avg_p)
    return opt_costs, avg_prices

print("Precomputing OPT costs and average accepted prices for competitive-ratio loss...")
opt_costs_all, opt_avgp_all = precompute_opt_costs_flex(
    price_all, base_all, flex_all, Delta_all, T, gamma, delta, c_delivery, eps_delivery, S
)
num_opt_ok = sum(1 for v in (opt_costs_all or []) if (v is not None and v > 1e-6))
print(f"OPT costs available for {num_opt_ok}/{total_instances} instances.")

# ---------------------------------------
# Build per-instance warm-up target y0 values
# ---------------------------------------
def compute_quantile(seq, q):
    if not seq:
        return None
    xs = sorted(float(v) for v in seq)
    idx = min(len(xs)-1, max(0, int(q * (len(xs)-1))))
    return xs[idx]

base_y0_targets = []
flexp_y0_targets = []
flexd_y0_targets = []
for i in range(total_instances):
    p_seq = price_all[i] if i < len(price_all) else []
    opt_avg = opt_avgp_all[i] if (opt_avgp_all and i < len(opt_avgp_all)) else None
    # Base and flex purchase share similar y space
    if opt_avg is not None and opt_avg > 0:
        tgt = float(opt_avg) + float(args.y0_margin)
    else:
        # fallback to a mid-high quantile of the instance price
        qv = compute_quantile(p_seq, 0.65) or float(p_min)
        tgt = float(qv)
    # clip to [p_min, p_max]
    tgt = max(float(p_min), min(float(p_max), tgt))
    base_y0_targets.append(tgt)
    flexp_y0_targets.append(tgt)
    flexd_y0_targets.append(tgt*(c_delivery + eps_delivery))  # flex delivery y0 target scaled

# random_base_targets, random_flexp_targets, random_flexd_targets = 

# Instantiate contextual model
feature_dim = 11  # handcrafted features length (see build_driver_features)
model = ThresholdPredictor(input_dim=feature_dim, K=K+1, hidden_dims=(64, 64))
model_device = torch.device("cpu")
model.to(model_device)
model.train()

# Initialize top-gate biases toward mean OPT accepted price (if available)
try:
    valid_avgps = [v for v in (opt_avgp_all or []) if (v is not None and v > 0)]
    if valid_avgps:
        opt_avgp_mean = float(sum(valid_avgps) / len(valid_avgps))
        # Map to fraction of [p_min, p_max]
        denom = max(1e-8, float(p_max) - float(p_min))
        frac = max(0.05, min(0.95, (opt_avgp_mean - float(p_min)) / denom))
        # logit
        bias_val = math.log(frac / (1.0 - frac))
        with torch.no_grad():
            for layer in [model.base_head_top, model.flex_p_head_top, model.flex_d_head_top]:
                layer.bias.fill_(bias_val)
        print(f"[init] Calibrated top-gate bias to target frac={frac:.3f} (bias={bias_val:.3f}) using OPT_avg_p≈{opt_avgp_mean:.2f}")
except Exception as e:
    print(f"[init] Top-gate calibration skipped: {e}")

## Feature builder per time step and driver kind
def build_driver_features(t_idx: int,
                          T: int,
                          price_seq: list[float],
                          time_seq: list[datetime.time],
                          month_seq: list[int],
                          forecast_seq: list[float],
                          storage_state: float,
                          kind: str,
                          b_or_f: float,
                          delta_idx: int,
                          p_min: float,
                          p_max: float) -> torch.Tensor:
    # Time features
    time = time_seq[t_idx]
    # convert time to a float hour + minute/60
    tau = time.hour + time.minute / 60.0 # this could be more effective as a sinusoid eventually
    rem = (T - 1 - t_idx) / max(T - 1, 1)
    # Month feature
    month_feat = (month_seq[t_idx] - 1) / 11.0 if month_seq and t_idx < len(month_seq) else 0.0
    # Forecasted price stats over remaining horizon
    rem_prices = forecast_seq[t_idx:] if t_idx < len(forecast_seq) else forecast_seq[-1:]
    if len(rem_prices) == 0:
        rem_prices = price_seq[-1:]
    p_mean = float(sum(rem_prices) / len(rem_prices))
    p_var = float(sum((x - p_mean) ** 2 for x in rem_prices) / max(1, len(rem_prices) - 1)) if len(rem_prices) > 1 else 0.0
    p_std = p_var ** 0.5
    p_min_rem = float(min(rem_prices))
    p_max_rem = float(max(rem_prices))
    feats = [
        tau,
        rem,
        month_feat,
        p_mean,
        p_var,
        p_std,
        p_min_rem,
        p_max_rem,
        price_seq[t_idx],
        float(storage_state),
        1.0,  # bias
    ]
    return torch.tensor(feats, dtype=torch.float32, device=model_device)

## CVX layers (base, flex purchase, flex delivery)
pald_base_layer = make_pald_base_layer(K)
pald_flex_purchase_layer = make_pald_flex_purchase_layer(K)
pald_flex_delivery_layer = make_pald_flex_delivery_layer(K)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Learning-rate scheduler intentionally disabled for current experiments

## Track the best model seen during training (by total loss)
best_snapshot = {
    "loss": float("inf"),
    "epoch": -1,
    "batch": -1,
    "model_state": None,
}

def _safe_layer_call(layer, y, args, size=1.0):
    """
    Call a CvxpyLayer and catch SCS/diffcp failures. Returns x_total tensor.
    """
    try:
        (x_total,) = layer(*args, solver_args=solver_options)
        return x_total
    except Exception as e:
        print(f"[warning] CvxpyLayer call failed: {e}")
        print("Current y:", [float(v) for v in y.detach().cpu().reshape(-1)])
        print("Current args:", args)
        exit(1)


def _group_model_params_by_name(model: ThresholdPredictor) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
    """Group parameters into trunk and head subgroups for diagnostics."""
    groups: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {
        "trunk": [],
        "base_logits": [], "base_top": [], "base_dec": [],
        "flexp_logits": [], "flexp_top": [], "flexp_dec": [],
        "flexd_logits": [], "flexd_top": [], "flexd_dec": [],
    }
    for name, p in model.named_parameters():
        if "trunk" in name:
            groups["trunk"].append((name, p))
        elif "base_head_logits" in name:
            groups["base_logits"].append((name, p))
        elif "base_head_top" in name:
            groups["base_top"].append((name, p))
        elif "base_head_dec" in name:
            groups["base_dec"].append((name, p))
        elif "flex_p_head_logits" in name or "flexp_head_logits" in name:
            groups["flexp_logits"].append((name, p))
        elif "flex_p_head_top" in name or "flexp_head_top" in name:
            groups["flexp_top"].append((name, p))
        elif "flex_p_head_dec" in name or "flexp_head_dec" in name:
            groups["flexp_dec"].append((name, p))
        elif "flex_d_head_logits" in name or "flexd_head_logits" in name:
            groups["flexd_logits"].append((name, p))
        elif "flex_d_head_top" in name or "flexd_head_top" in name:
            groups["flexd_top"].append((name, p))
        elif "flex_d_head_dec" in name or "flexd_head_dec" in name:
            groups["flexd_dec"].append((name, p))
        else:
            # default to trunk if unknown
            groups["trunk"].append((name, p))
    return groups


def _param_tensor_norm(params: List[torch.nn.Parameter]) -> float:
    acc = 0.0
    for p in params:
        if p is None:
            continue
        acc += float(p.detach().data.float().norm(2).item() ** 2)
    return acc ** 0.5


def _grad_tensor_norm(params: List[torch.nn.Parameter]) -> float:
    acc = 0.0
    for p in params:
        if p is None or p.grad is None:
            continue
        g = p.grad.detach().data.float()
        acc += float(g.norm(2).item() ** 2)
    return acc ** 0.5


def _collect_prev_params(model: ThresholdPredictor) -> Dict[str, torch.Tensor]:
    prev: Dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        prev[name] = p.detach().cpu().clone()
    return prev


def _update_norm_since(prev: Dict[str, torch.Tensor], model: ThresholdPredictor, names: List[str]) -> Tuple[float, float]:
    """Return (delta_norm, current_norm) across the given parameter names."""
    delta_sq = 0.0
    curr_sq = 0.0
    for name, p in model.named_parameters():
        if name not in prev:
            continue
        if any(name == n for n in names):
            curr = p.detach().cpu()
            pr = prev[name]
            d = curr - pr
            delta_sq += float(d.float().pow(2).sum().item())
            curr_sq += float(curr.float().pow(2).sum().item())
    return (delta_sq ** 0.5, curr_sq ** 0.5)


def _sample_gate_stats(model: ThresholdPredictor, sample_n: int,
                       price_all: list, times_all: list, months_all: list, forecast_all: list,
                       p_min: float, p_max: float, device) -> Dict[str, Tuple[float, float, float]]:
    """Return mean, p25, p75 of sigmoid(top) and sigmoid(dec) for each head over a small sample.
    Keys: base_top, base_dec, flexp_top, flexp_dec, flexd_top, flexd_dec.
    """
    idxs = list(range(min(sample_n, len(price_all))))
    vals = {k: [] for k in ["base_top", "flexp_top", "flexd_top"]}
    with torch.no_grad():
        for i in idxs:
            feats = build_driver_features(0, T, price_all[i], times_all[i], months_all[i], forecast_all[i], 0.0, "base", S, 0, float(p_min), float(p_max)).to(device)
            try:
                h = model.trunk(feats.unsqueeze(0))  # (1, H)
                bt = torch.sigmoid(model.base_head_top(h))
                vals["base_top"].append(float(bt.item()))
            except Exception:
                pass
            try:
                h = model.trunk(feats.unsqueeze(0))
                ft = torch.sigmoid(model.flex_p_head_top(h))
                vals["flexp_top"].append(float(ft.item()))
            except Exception:
                pass
            try:
                h = model.trunk(feats.unsqueeze(0))
                dt = torch.sigmoid(model.flex_d_head_top(h))
                vals["flexd_top"].append(float(dt.item()))
            except Exception:
                pass
    import numpy as np
    stats: Dict[str, Tuple[float, float, float]] = {}
    for k, arr in vals.items():
        if not arr:
            stats[k] = (float('nan'), float('nan'), float('nan'))
        else:
            a = np.array(arr)
            stats[k] = (float(a.mean()), float(np.percentile(a, 25)), float(np.percentile(a, 75)))
    return stats


## Main training loop
losses = []
try: 
    for epoch in range(epochs):
    # Snapshot params to measure actual parameter movement during this epoch
        prev_params_snapshot = _collect_prev_params(model)
        # Collect CRs for the last processed batch to print in summary
        last_batch_crs = []
        # Simple stagnation diagnostics
        last_gnorm = 0.0
        clip_events = 0
        # Track last computed loss this epoch (last batch)
        last_epoch_loss = float('nan')

        # Enable/disable parameter groups depending on warm-up (only the boolean; per-param freezing remains off)
        warmup_active = epoch < int(args.warmup_epochs)
        # if warmup_active:
        #     # Freeze everything except the top gate biases/weights to shape y0
        #     for name, p in model.named_parameters():
        #         if any(k in name for k in ["_head_top.weight", "_head_top.bias"]):
        #             p.requires_grad_(True)
        #         else:
        #             p.requires_grad_(False)
        # else:
        #     # Immediately after warm-up, freeze trunk for a few epochs to stabilize heads
        #     post_warm = epoch - int(args.warmup_epochs)
        #     for name, p in model.named_parameters():
        #         if post_warm >= 0 and post_warm < int(args.freeze_trunk_epochs):
        #             if any(k in name for k in ["trunk."]):
        #                 p.requires_grad_(False)
        #             else:
        #                 p.requires_grad_(True)
        #         else:
        #             p.requires_grad_(True)

        # Diagnostics accumulators per epoch
        epoch_base_calls = 0
        epoch_base_active = 0
        epoch_flex_calls = 0
        epoch_flex_active = 0
        epoch_forced_topup_sum = 0.0
        epoch_forced_topup_events = 0
        epoch_xpre_zero = 0
        epoch_xpre_count = 0
        epoch_y0_base_sum = 0.0
        epoch_y0_fp_sum = 0.0
        epoch_y0_fd_sum = 0.0
        epoch_y2_base_sum = 0.0  # second-to-last element averages
        epoch_y2_fp_sum = 0.0
        epoch_y2_fd_sum = 0.0
        epoch_inst_count = 0
        
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            price_batch = price_all[start:end]
            time_batch = times_all[start:end]
            month_batch = months_all[start:end]
            forecast_batch = forecast_all[start:end]
            base_batch = base_all[start:end]
            flex_batch = flex_all[start:end]
            Delta_batch = Delta_all[start:end]
            batch_total_loss = torch.tensor(0.0)
            warmup_loss_sum = torch.tensor(0.0)

            batch_crs = []  # competitive ratios for this batch
            # Iterate instances in this batch
            for idx, (price_seq, time_seq, month_seq, forecast_seq, base_seq, flex_seq, Delta_seq) in enumerate(zip(price_batch, time_batch, month_batch, forecast_batch, base_batch, flex_batch, Delta_batch)):
                global_idx = start + idx  # align with precomputed OPT list
                # Get representative y0 predictions for this instance (t=0, zero storage)
                try:
                    feats_b = build_driver_features(0, T, price_seq, time_seq, month_seq, forecast_seq, 0.0, "base", S, 0, float(p_min), float(p_max))
                    yb, yfp, yfd = model(feats_b, p_min=float(p_min), p_max=float(p_max))
                    epoch_y0_base_sum += float(yb[0].detach().item())
                    epoch_y0_fp_sum += float(yfp[0].detach().item())
                    epoch_y0_fd_sum += float(yfd[0].detach().item())
                    if yb.shape[0] >= 2:
                        epoch_y2_base_sum += float(yb[-2].detach().item())
                        epoch_y2_fp_sum += float(yfp[-2].detach().item())
                        epoch_y2_fd_sum += float(yfd[-2].detach().item())
                except Exception:
                    pass
                epoch_inst_count += 1
                # global storage state and decision both start at 0
                storage_state = torch.tensor(0.0, dtype=torch.float32)
                x_prev_global = torch.tensor(0.0)

                # Each base driver tracks fractional progress (unit capacity); demand scales the fractional decision
                base_drivers = []  # list of dicts with keys: id, b (demand), w (fraction), prev_decision (fraction)
                # Predict threshold for this base driver
                feats = build_driver_features(
                    t_idx=0, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                    storage_state=S,
                    kind="base", b_or_f=S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                )
                y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                # log y_vec_t to the console
                # print(f"[debug] Initial base y_vec: {[float(v) for v in y_vec_t.detach().cpu()]}")
                base_drivers.append({
                    "id": 0,
                    "b": S,
                    "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1),
                    "prev_decision": torch.tensor(0.0, dtype=torch.float32).reshape(-1),
                    "y_vec": y_vec_t,
                })
                # Flexible drivers: track purchase (w) and delivery (v) progress fractions
                flex_drivers = []  # dict keys: id, f, delta, w, v, prev_x, prev_z

                # collect decisions so far
                x_hist = []
                z_hist = []

                for t in range(T):
                    b_t_val = float(base_seq[t])
                    p_t_val = float(price_seq[t])

                    # Add flexible driver arrivals if non-zero
                    f_arrival = float(flex_seq[t])
                    dlt = int(Delta_seq[t])
                    if f_arrival > 0:
                        feats = build_driver_features(
                            t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                            storage_state=S,
                            kind="flex", b_or_f=f_arrival, delta_idx=dlt, p_min=float(p_min), p_max=float(p_max)
                        )
                        _, y_vec_p, y_vec_d = model(feats, p_min=float(p_min), p_max=float(p_max))
                        flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": dlt, 
                                             "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                             "v": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                             "prev_x": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                             "prev_z": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                             "y_vec_purchase": y_vec_p, "y_vec_delivery": y_vec_d})

                    # Add base driver if non-zero
                    if b_t_val > 0:
                        # if the base demand is larger than S, we can just refresh the base drivers and add a single driver with size S
                        if b_t_val > S:
                            # print(f"[warning] base demand {b_t_val} at t={t} exceeds S={S}, capping to S")
                            base_drivers = [] # reset previous base drivers
                            feats = build_driver_features(
                                t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                                storage_state=S,
                                kind="base", b_or_f=S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                            )
                            y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                            base_drivers.append({"id": 0, "b": S, 
                                                 "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                                 "prev_decision": torch.tensor(0.0, dtype=torch.float32).reshape(-1), "y_vec": y_vec_t})
                        else:
                            # Predict threshold for this base driver
                            feats = build_driver_features(
                                t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                                storage_state=S,
                                kind="base", b_or_f=b_t_val, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                            )
                            y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                            base_drivers.append({"id": 2 * t + 2, "b": b_t_val, 
                                                 "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                                 "prev_decision": torch.tensor(0.0, dtype=torch.float32).reshape(-1), "y_vec": y_vec_t})

                    # Compute purchasing excess from previous step in physical units
                    prev_purchasing_total = torch.tensor(0.0, dtype=torch.float32)
                    for drv in base_drivers:
                        prev_purchasing_total = prev_purchasing_total + (drv["prev_decision"] * drv["b"])  # tensor
                    for fd in flex_drivers:
                        prev_purchasing_total = prev_purchasing_total + (fd["prev_x"] * fd["f"])  # tensor
                    # Keep x_prev_global as tensor for gradient flow across time
                    purchasing_excess = x_prev_global - prev_purchasing_total

                    # Compute delivery excess from previous step in physical units
                    prev_delivery_total = torch.tensor(0.0, dtype=torch.float32)
                    for fd in flex_drivers:
                        prev_delivery_total = prev_delivery_total + (fd["prev_z"] * fd["f"])  # tensor
                    # last z was base b_{t-1} + flex deliveries; but we only need per-driver shares here

                    # compute the cumulative upper bound on the buying decision at the current time step:
                    # this buy cap is (S - storage_state) + possible z_t
                    
                    # first determine the flex deliveries

                    # Base delivery equals current base demand arrival
                    z_components = [torch.tensor(b_t_val, dtype=torch.float32).reshape(-1)]

                    # Flexible drivers: delivery decisions
                    for fd in flex_drivers:
                        f_i = fd["f"]
                        prev_frac_z = fd["prev_z"]
                        y_vec_d = fd["y_vec_delivery"]
                        # share positive excess proportional to previous physical contribution
                        # share delivery excess (if you track it globally); here we just use prev_frac_z
                        v_prev_frac = fd["v"]
                        w_prev_frac = fd["w"]

                        # Enforce deadline and purchase cap outside the layer (keeps DPP)
                        if t >= max(0, int(fd["delta"])):
                            # need to deliver remainder
                            cur_frac_z = torch.clamp(1.0 - v_prev_frac, min=0.0).reshape(-1)
                            z_components.append(cur_frac_z)
                            fd["prev_z"] = cur_frac_z.reshape(-1)
                            fd["v"] = (fd["v"] + cur_frac_z).reshape(-1)
                            continue
                        # if w is really low, just force a zero decision
                        if float(w_prev_frac.detach().item()) <= 1e-9:
                            # no more buying possible, force zero decision
                            cur_frac_z = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            z_components.append(cur_frac_z)
                            fd["prev_z"] = cur_frac_z.reshape(-1)
                            fd["v"] = (fd["v"] + cur_frac_z).reshape(-1)
                            continue
                        else:
                            # clamp v into [0, 1 - eps] to avoid issues with the solver
                            v_eff = torch.clamp(v_prev_frac, max=1.0 - 1e-9)
                            if (1.0 - float(v_eff.detach().item())) <= 1e-9:
                                cur_frac_z = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                                cur_frac_z = torch.clamp(cur_frac_z, max=max(0.0, w_prev_frac - v_prev_frac)).reshape(-1)
                                z_components.append(cur_frac_z)
                                fd["prev_z"] = cur_frac_z
                                fd["v"] = (fd["v"] + cur_frac_z).reshape(-1)
                                continue
                            # z_prev_clamped = torch.clamp(prev_frac_z, max=1.0 - 1e-9)
                            y_vec_d = fd["y_vec_delivery"]
                            p_t_t = torch.tensor([p_t_val], dtype=torch.float32)
                            delta_t = torch.tensor(delta, dtype=torch.float32).reshape(-1)

                            # Precompute coeff = p_t * (c+eps) - p_t * c * s_prev  (scalar)
                            coeff_t = p_t_t * (torch.tensor(c_delivery + eps_delivery) - torch.tensor(c_delivery) * storage_state)
                                                            
                            # precompute hinge
                            w_hinge_t, c1_t = hinge_from_y_torch(taus_full_t, y_vec_d)

                            cur_frac_z = _safe_layer_call(
                                pald_flex_delivery_layer, y_vec_d, (fd["prev_z"], v_eff, coeff_t, delta_t, w_hinge_t, c1_t)
                            )
                        cur_phys_z = torch.mul(cur_frac_z, f_i).reshape(-1)
                        z_components.append(cur_phys_z)

                        # Update state for the next step (kept differentiable)
                        fd["prev_z"] = cur_frac_z.reshape(-1)
                        fd["v"] = (fd["v"] + cur_frac_z).reshape(-1)

                    z_t = torch.stack(z_components).sum()

                    # now that we have the delivery z_t, we can compute the buy cap
                    buy_cap_t = torch.tensor(S, dtype=torch.float32) - storage_state + z_t
                    # buy_cap_t = torch.tensor(buy_cap, dtype=torch.float32)
                    # we will decrement from this buy_cap as we allocate to drivers below

                    # Determine per-driver decisions (fractional)
                    decisions = []  # list of tensors in physical units
                    for drv in base_drivers:
                        if float(buy_cap_t.detach().item()) <= 1e-9:
                            # no more buying possible, force zero decision
                            cur_phys_decision = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            decisions.append(cur_phys_decision)
                            drv["prev_decision"] = cur_phys_decision
                            drv["w"] = (drv["w"] + drv["prev_decision"]).reshape(-1)
                            continue

                        b_i = drv["b"]
                        prev_frac = drv["prev_decision"]
                        y_vec_t = drv["y_vec"]

                        # share positive excess proportional to previous physical contribution
                        # Compute share and pseudo previous fraction with tensor ops (keeps autograd)
                        denom_safe = torch.clamp(prev_purchasing_total, min=1e-8)
                        share = torch.where(prev_purchasing_total > 1e-12,
                                            (prev_frac * b_i) / denom_safe,
                                            torch.tensor(0.0, dtype=torch.float32))
                        positive_excess = torch.clamp(purchasing_excess, min=0.0)
                        pseudo_prev_frac = prev_frac + positive_excess * share / max(b_i, 1e-8)

                        w_prev_frac = drv["w"]
                        # Clamp w into [0, 1 - eps] to avoid issues with the solver
                        w_eff = torch.clamp(w_prev_frac, max=1.0 - 1e-9)
                        if (1.0 - float(w_eff.detach().item())) <= 1e-9:
                            cur_frac_decision = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            decisions.append(cur_frac_decision)
                            drv["prev_decision"] = cur_frac_decision
                            drv["w"] = (drv["w"] + drv["prev_decision"]).reshape(-1)
                            continue

                        # === forward ===
                        w_hinge_t, c1_t = hinge_from_y_torch(taus_full_t, y_vec_t)

                        gamma_t = torch.tensor(gamma, dtype=torch.float32).reshape(-1)
                        p_t_t = torch.tensor([p_t_val], dtype=torch.float32)

                        cur_frac_decision = _safe_layer_call(
                            pald_base_layer, y_vec_t, (pseudo_prev_frac, w_eff, p_t_t, gamma_t, w_hinge_t, c1_t)
                        )

                        # diagnostics: base activation
                        epoch_base_calls += 1
                        if float(cur_frac_decision.detach().item()) > 1e-9:
                            epoch_base_active += 1

                        # Convert to physical units by scaling with demand of this driver
                        cur_phys_decision = torch.mul(cur_frac_decision, b_i).reshape(-1)

                        # check if this decision exceeds the remaining buy cap (use scalar check for control flow)
                        if float((cur_phys_decision - buy_cap_t).detach().item()) > 1e-5:
                            # take the remaining buy cap instead
                            cur_phys_decision = (buy_cap_t.to(torch.float32)).reshape(-1)
                            # and set the fractional decision accordingly
                            if b_i > 1e-8:
                                cur_frac_decision = cur_phys_decision / b_i
                            else:
                                cur_frac_decision = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            # after this, the buy cap is zero
                            buy_cap_t = torch.tensor(0.0, dtype=torch.float32)

                        decisions.append(cur_phys_decision)
                        buy_cap_t = buy_cap_t - cur_phys_decision

                        # Update state for the next step (kept differentiable)
                        drv["w"] = (drv["w"] + cur_frac_decision).reshape(-1)
                        drv["prev_decision"] = (cur_frac_decision).reshape(-1)

                    # Flexible drivers: purchasing decisions
                    for fd in flex_drivers:
                        if float(buy_cap_t.detach().item()) <= 1e-9:
                            # no more buying possible, force zero decision
                            cur_phys_x = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            decisions.append(cur_phys_x)
                            fd["prev_x"] = cur_phys_x
                            fd["w"] = (fd["w"] + fd["prev_x"]).reshape(-1)
                            continue

                        f_i = fd["f"]
                        prev_frac_x = fd["prev_x"]
                        y_vec_t = fd["y_vec_purchase"]

                        denom_safe = torch.clamp(prev_purchasing_total, min=1e-8)
                        share = torch.where(prev_purchasing_total > 1e-12,
                                            (prev_frac_x * f_i) / denom_safe,
                                            torch.tensor(0.0, dtype=torch.float32))
                        positive_excess = torch.clamp(purchasing_excess, min=0.0)
                        pseudo_prev_x = prev_frac_x + positive_excess * share / max(f_i, 1e-8)

                        w_prev_frac = fd["w"]
                        # Clamp w into [0, 1 - eps] to avoid issues with the solver
                        w_eff = torch.clamp(w_prev_frac, max=1.0 - 1e-9)
                        if (1.0 - float(w_eff.detach().item())) <= 1e-9:
                            cur_frac_x = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            decisions.append(cur_frac_x)
                            fd["prev_x"] = cur_frac_x
                            fd["w"] = (fd["w"] + fd["prev_x"]).reshape(-1)
                            continue

                        # Enforce deadline and purchase cap outside the layer (keeps DPP)
                        if t >= max(0, int(fd["delta"])):
                            # need to buy remainder
                            cur_frac_x = torch.clamp(1.0 - w_prev_frac, min=0.0).reshape(-1)
                            decisions.append(cur_frac_x)
                            fd["prev_x"] = cur_frac_x
                            fd["w"] = (fd["w"] + fd["prev_x"]).reshape(-1)
                            cur_phys_x = torch.mul(cur_frac_x, f_i).reshape(-1)
                            buy_cap_t = buy_cap_t - cur_phys_x
                            continue
                        else:
                            # x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_x)))

                            # === forward ===
                            w_hinge_t, c1_t = hinge_from_y_torch(taus_full_t, y_vec_t)

                            gamma_t = torch.tensor(gamma, dtype=torch.float32).reshape(-1)
                            p_t_t = torch.tensor([p_t_val], dtype=torch.float32)

                            cur_frac_x = _safe_layer_call(
                                pald_flex_purchase_layer, y_vec_t, (pseudo_prev_x, w_eff, p_t_t, gamma_t, w_hinge_t, c1_t)
                            )
                        # diagnostics: flex purchase activation
                        epoch_flex_calls += 1
                        if float(cur_frac_x.detach().item()) > 1e-9:
                            epoch_flex_active += 1
                        
                        cur_phys_x = torch.mul(cur_frac_x, f_i).reshape(-1)

                        # check if this decision exceeds the remaining buy cap
                        if float((cur_phys_x - buy_cap_t).detach().item()) > 1e-5:
                            # take the remaining buy cap instead
                            cur_phys_x = (buy_cap_t.to(torch.float32)).reshape(-1)
                            # and set the fractional decision accordingly
                            if f_i > 1e-8:
                                cur_frac_x = cur_phys_x / f_i
                            else:
                                cur_frac_x = torch.tensor(0.0, dtype=torch.float32).reshape(-1)
                            # after this, the buy cap is zero
                            buy_cap_t = torch.tensor(0.0, dtype=torch.float32)
                        
                        decisions.append(cur_phys_x)
                        buy_cap_t = buy_cap_t - cur_phys_x

                        # Update state for the next step (kept differentiable)
                        fd["w"] = (fd["w"] + cur_frac_x).reshape(-1)
                        fd["prev_x"] = (cur_frac_x).reshape(-1)

                    # Aggregate physical purchases this step
                    # print decisions to debug any differences in shape
                    x_t = torch.stack(decisions).sum() if decisions else torch.tensor(0.0)

                    # Diagnostics: forced top-up outside CVX layers
                    x_pre_val = float(x_t.detach().item())
                    
                    # Ensure purchases cover deliveries (inventory feasibility)
                    # storage_state is maintained as a torch scalar throughout
                    required = torch.maximum(z_t - storage_state, torch.tensor(0.0, dtype=z_t.dtype))
                    forced_extra = torch.maximum(required - torch.tensor(x_pre_val, dtype=torch.float32), torch.tensor(0.0, dtype=z_t.dtype))
                    if forced_extra > 1e-9:
                        epoch_forced_topup_sum += forced_extra.item()
                        epoch_forced_topup_events += 1
                    epoch_xpre_count += 1
                    if x_pre_val <= 1e-9:
                        epoch_xpre_zero += 1
                    x_t = torch.maximum(x_t, z_t - storage_state)

                    # diagnostics -- check if the currect decision will ``overfill the storage''
                    if float(storage_state.detach().item() + x_t.detach().item() - z_t.detach().item()) > S + 1e-3:
                        print(f"[warning] t={t} overfill: storage {storage_state:.3f} + x {float(x_t.detach()):.3f} - z {float(z_t.detach()):.3f} > S={S}")
                    # Track previous storage (for refresh condition), then update differentiably
                    prev_storage_scalar = float(storage_state.detach().item())
                    storage_state = torch.clamp(storage_state + x_t - z_t, min=0.0, max=S)
                    # Propagate previous x as tensor (no detach)
                    x_prev_global = x_t

                    # if the storage will be empty and it was previously non-empty, we can refresh the base drivers
                    s_now = float(storage_state.detach().item())
                    s_prev = prev_storage_scalar
                    if s_now <= 1e-9 and s_prev > 1e-9:
                        # print(f"[info] t={t} storage emptied, refreshing base drivers")
                        base_drivers = []  # reset previous base drivers
                        feats = build_driver_features(
                            t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                            storage_state=S,
                            kind="base", b_or_f=S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                        )
                        y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                        base_drivers.append({"id": 0, "b": S, 
                                             "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                             "prev_decision": torch.tensor(0.0, dtype=torch.float32).reshape(-1), "y_vec": y_vec_t})
                    # storage_state already clamped    
                    
                    # record sequences for torch objective
                    x_hist.append(x_t)
                    z_hist.append(z_t)

                # Convert sequences for torch objective
                p_torch = torch.tensor([float(v) for v in price_seq], dtype=torch.float32)
                x_torch = torch.stack(x_hist) if x_hist else torch.ones(T)
                z_torch = torch.stack(z_hist) if z_hist else torch.zeros(T)

                pald_cost = torch_objective(p_torch, x_torch, z_torch, gamma, delta, c_delivery, eps_delivery)
                # Collect CR for this instance if OPT exists
                try:
                    opt_val = opt_costs_all[global_idx] if opt_costs_all is not None else None
                    if opt_val is not None and opt_val > 1e-9:
                        cr_val = float(pald_cost.item()) / float(opt_val)
                        if cr_val < 1.0:
                            cr_val = 1.0  # numerical issues
                            # for debugging, if the competitive ratio is less than one, print the following:
                            # the total amount of demand (sum of base and flex)
                            total_demand = sum(base_seq) + sum(flex_seq)
                            print(f"[warning] pald cost {pald_cost.item():.3f} < opt {opt_val:.3f}, total demand {total_demand}, setting CR=1.0")
                            # also print the amount delivered by PALD
                            print(f"  total delivered: {float(torch.sum(z_torch).item()):.3f}")
                            # print the purchasing sequence
                            print(f"  purchasing: {[float(v.detach().item()) for v in x_torch]}")
                            # print the delivery sequence
                            print(f"  delivery: {[float(v.detach().item()) for v in z_torch]}")
                        batch_crs.append(cr_val)
                except Exception:
                    pass

                if not use_cost_loss:
                    # # Competitive-ratio loss: ReLU(pald/opt - 1), opt treated as constant
                    inst_loss = torch.tensor(0.0)
                    if opt_costs_all is not None:
                        opt_val = opt_costs_all[global_idx]
                        if opt_val is not None and opt_val > 1e-6:
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

                # Warm-up loss: pull y0 for base and flex purchase toward targets
                if warmup_active:
                    try:
                        # y0 targets for this instance
                        y0_tgt = float(base_y0_targets[global_idx])
                        y0_tgt_flex = float(flexp_y0_targets[global_idx])
                        y0_tgt_flexd = float(flexd_y0_targets[global_idx])
                        # Build features at t=0 with zero storage for consistency
                        feats_b0 = build_driver_features(0, T, price_seq, time_seq, month_seq, forecast_seq, 0.0, "base", S, 0, float(p_min), float(p_max))
                        yb_all, yfp_all, yd_all = model(feats_b0, p_min=float(p_min), p_max=float(p_max))
                        yb0 = yb_all[0]
                        yfp0 = yfp_all[0]
                        yd0 = yd_all[0]
                        # L1 warm-up encourages moving the top without over-penalizing outliers
                        y0_tgt_t = torch.tensor(y0_tgt, dtype=torch.float32)
                        y0_tgt_flex_t = torch.tensor(y0_tgt_flex, dtype=torch.float32)
                        y0_tgt_flexd_t = torch.tensor(y0_tgt_flexd, dtype=torch.float32)
                        warmup_inst = torch.abs(yb0 - y0_tgt_t) + torch.abs(yfp0 - y0_tgt_flex_t) + torch.abs(yd0 - y0_tgt_flexd_t)
                        warmup_loss_sum = warmup_loss_sum + warmup_inst
                    except Exception:
                        pass
                

            # end for each instance

            # Normalize by batch size
            batch_total_loss = batch_total_loss / batch_size
            # Save CRs for summary printing
            last_batch_crs = batch_crs

            # Smoothness regularization on a sample predicted vector: encourage gradual thresholds
            shape_penalty = torch.tensor(0.0)
            if len(price_batch) > 0:
                feats_s = build_driver_features(0, T, price_batch[0], time_batch[0], month_batch[0], forecast_batch[0], 0.0, "base", S, 0, float(p_min), float(p_max))
                yb, yp, yd = model(feats_s, p_min=float(p_min), p_max=float(p_max))
                def smoothness(y):
                    dif = y[:-1] - y[1:]  # non-negative by design
                    return (dif * dif).sum()
                shape_penalty = 0.01 * smoothness(yb) + 0.005 * smoothness(yp) + 0.005 * smoothness(yd)

            # Final loss: CR/cost loss + small shape penalty + warm-up (if active) + top-up penalty + post-warmup anchor
            if warmup_active:
                # Average warm-up over batch
                warmup_term = (warmup_loss_sum / max(1, batch_size)) * float(args.warmup_lambda)
            else:
                warmup_term = torch.tensor(0.0)
            topup_term = torch.tensor(float(epoch_forced_topup_sum), dtype=torch.float32) * float(args.topup_penalty_lambda)
            # Post-warmup anchor: decays linearly to 0 over post_warmup_epochs
            if not warmup_active and int(args.post_warmup_epochs) > 0:
                post_warm = max(0, epoch - int(args.warmup_epochs))
                decay = max(0.0, 1.0 - (post_warm / float(args.post_warmup_epochs)))
                anchor_w = float(args.post_warmup_lambda) * decay
                # Use same y0 targets as warm-up; only include base and flex purchase heads
                # Compute average anchor over this batch
                anchor_sum = torch.tensor(0.0)
                count = 0
                for idx, price_seq in enumerate(price_batch):
                    global_idx = start + idx
                    try:
                        y0_tgt = float(base_y0_targets[global_idx])
                        feats_b0 = build_driver_features(0, T, price_seq, time_batch[idx], month_batch[idx], forecast_batch[idx], 0.0, "base", S, 0, float(p_min), float(p_max))
                        yb_all, yfp_all, _ = model(feats_b0, p_min=float(p_min), p_max=float(p_max))
                        yb0 = yb_all[0]
                        yfp0 = yfp_all[0]
                        y0_tgt_t = torch.tensor(y0_tgt, dtype=torch.float32)
                        anchor_sum = anchor_sum + (torch.abs(yb0 - y0_tgt_t) + torch.abs(yfp0 - y0_tgt_t))
                        count += 1
                    except Exception:
                        pass
                anchor_term = (anchor_sum / max(1, count)) * anchor_w
            else:
                anchor_term = torch.tensor(0.0)
            loss = batch_total_loss + shape_penalty + warmup_term + topup_term + anchor_term
            losses.append(float(loss.item()))
            last_epoch_loss = float(loss.item())

            optimizer.zero_grad()
            loss.backward()
            gnorm = clip_grad_norm_(model.parameters(), max_norm=1.0)
            try:
                last_gnorm = float(gnorm)
                if last_gnorm > 1.0:
                    clip_events += 1
            except Exception:
                pass
            optimizer.step()
            # scheduler stepping disabled

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
                # gradient norms (sanity check)
                total_norm = 0.0
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += float(p.grad.data.norm(2).item())
                print(f"epoch {epoch} batch {batch_idx}, loss {loss.item():.4f} | grad_norm~ {total_norm:.3e}")

    # Epoch summary (every 10 epochs or last)
        if (epoch % 10 == 0) or (epoch == epochs - 1):
            with torch.no_grad():
                base_act_rate = (epoch_base_active / max(1, epoch_base_calls)) if epoch_base_calls else 0.0
                flex_act_rate = (epoch_flex_active / max(1, epoch_flex_calls)) if epoch_flex_calls else 0.0
                xpre_zero_frac = (epoch_xpre_zero / max(1, epoch_xpre_count)) if epoch_xpre_count else 0.0
                avg_forced = (epoch_forced_topup_sum / max(1, epoch_forced_topup_events)) if epoch_forced_topup_events else 0.0
                y0b = (epoch_y0_base_sum / max(1, epoch_inst_count)) if epoch_inst_count else 0.0
                y0fp = (epoch_y0_fp_sum / max(1, epoch_inst_count)) if epoch_inst_count else 0.0
                y0fd = (epoch_y0_fd_sum / max(1, epoch_inst_count)) if epoch_inst_count else 0.0
                y2b = (epoch_y2_base_sum / max(1, epoch_inst_count)) if epoch_inst_count else 0.0
                y2fp = (epoch_y2_fp_sum / max(1, epoch_inst_count)) if epoch_inst_count else 0.0
                y2fd = (epoch_y2_fd_sum / max(1, epoch_inst_count)) if epoch_inst_count else 0.0
                opt_avgp = float(sum(v for v in opt_avgp_all if (v is not None and v > 0))) / max(1, sum(1 for v in opt_avgp_all if (v is not None and v > 0))) if opt_avgp_all else float('nan')
                cr_line = " ".join(f"{v:.2f}" for v in last_batch_crs) if last_batch_crs else "N/A"
                # Parameter movement and gradient norms per group
                groups = _group_model_params_by_name(model)
                grad_info = {k: _grad_tensor_norm([p for _, p in v]) for k, v in groups.items()}
                # Update norm since epoch start for selected groups
                upd_trunk = _update_norm_since(prev_params_snapshot, model, [n for n, _ in groups.get('trunk', [])])
                upd_btop = _update_norm_since(prev_params_snapshot, model, [n for n, _ in groups.get('base_top', [])])
                upd_fptop = _update_norm_since(prev_params_snapshot, model, [n for n, _ in groups.get('flexp_top', [])])
                upd_fdtop = _update_norm_since(prev_params_snapshot, model, [n for n, _ in groups.get('flexd_top', [])])
                # Learning rate and trainable counts
                try:
                    curr_lr = float(optimizer.param_groups[0].get('lr', float('nan')))
                except Exception:
                    curr_lr = float('nan')
                trainable_counts = {k: sum(1 for _, p in groups.get(k, []) if p.requires_grad) for k in groups.keys()}
                # Sample gate saturation stats
                gate_stats = _sample_gate_stats(model, sample_n=min(16, len(price_all)),
                                                price_all=price_all, times_all=times_all, months_all=months_all, forecast_all=forecast_all,
                                                p_min=float(p_min), p_max=float(p_max), device=model_device)
                print(
                    f"[epoch {epoch} summary] loss={last_epoch_loss:.4f} "
                    f"act_base={base_act_rate:.3f} act_flex={flex_act_rate:.3f} xpre_zero={xpre_zero_frac:.3f} "
                    f"forced_topup_avg={avg_forced:.4f} "
                    f"y0_b={y0b:.2f} y0_fp={y0fp:.2f} y0_fd={y0fd:.2f} "
                    f"y2_b={y2b:.2f} y2_fp={y2fp:.2f} y2_fd={y2fd:.2f} "
                    f"OPT_avg_p={opt_avgp:.2f} | CRs: {cr_line}\n"
                    f"  grads[L2]: trunk={grad_info.get('trunk', 0.0):.2e} b_top={grad_info.get('base_top', 0.0):.2e} fp_top={grad_info.get('flexp_top', 0.0):.2e} fd_top={grad_info.get('flexd_top', 0.0):.2e}\n"
                    f"  upd_norm: trunk d={upd_trunk[0]:.2e}/||w||={upd_trunk[1]:.2e} | b_top d={upd_btop[0]:.2e} fp_top d={upd_fptop[0]:.2e} fd_top d={upd_fdtop[0]:.2e}\n"
                    f"  opt: lr={curr_lr:.2e} last_gnorm={last_gnorm:.2e} clipped_times={clip_events} trainable={sum(trainable_counts.values())}"
                    f"  gates (mean[p25,p75]): base_top={gate_stats['base_top'][0]:.2f}[{gate_stats['base_top'][1]:.2f},{gate_stats['base_top'][2]:.2f}]"
                    f" flexp_top={gate_stats['flexp_top'][0]:.2f}[{gate_stats['flexp_top'][1]:.2f},{gate_stats['flexp_top'][2]:.2f}]"
                    f" flexd_top={gate_stats['flexd_top'][0]:.2f}[{gate_stats['flexd_top'][1]:.2f},{gate_stats['flexd_top'][2]:.2f}]"
                )

            # After projection, update best snapshot if improved
            with torch.no_grad():
                curr = float(last_epoch_loss)
                if curr + 1e-12 < best_snapshot["loss"]:
                    best_snapshot["loss"] = curr
                    best_snapshot["epoch"] = epoch
                    best_snapshot["batch"] = batch_idx
                    best_snapshot["model_state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(f"[best] Updated at epoch {epoch} batch {batch_idx}: loss={curr:.6f}")
            # (no LR scheduler)

except KeyboardInterrupt:
    print("[train] Caught KeyboardInterrupt. Skipping remaining training...")

# save the model at the end of training
if best_snapshot["model_state"] is not None:
    try:
        os.makedirs("best_models", exist_ok=True)
        model_path = f"best_models/pald_model_{trace}_{month}_{batch_size}_{run_tag}.pt"
        torch.save(best_snapshot["model_state"], model_path)
        run_generated_files.append(model_path)
        print(f"Saved model: {model_path}")
    except Exception as e:
        print(f"Model save failed: {e}")

# Write a simple text log for this run
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
        if best_snapshot["model_state"] is not None:
            f.write("\nBest snapshot:\n")
            f.write(f"  loss: {best_snapshot['loss']:.6f}\n")
            f.write(f"  epoch: {best_snapshot['epoch']}\n")
            f.write(f"  batch: {best_snapshot['batch']}\n")
            # Log a representative set of thresholds by running the model on a sample feature vector
            try:
                sample_idx = 0 if total_instances > 0 else None
                if sample_idx is not None:
                    p_seq = price_all[sample_idx]
                    time_seq = times_all[sample_idx]
                    month_seq = months_all[sample_idx]
                    forecast_seq = forecast_all[sample_idx]
                    b_seq = base_all[sample_idx]
                    # Build a base-driver feature at t=0 with zero storage as a representative context
                    feats = build_driver_features(
                        t_idx=0,
                        T=T,
                        price_seq=p_seq,
                        time_seq=time_seq,
                        month_seq=month_seq,
                        forecast_seq=forecast_seq,
                        storage_state=0.0,
                        kind="base",
                        b_or_f=(float(b_seq[0]) if len(b_seq) > 0 else S),
                        delta_idx=0,
                        p_min=float(p_min),
                        p_max=float(p_max),
                    ).unsqueeze(0)
                    model.eval()
                    with torch.no_grad():
                        y_b, y_fp, y_fd = model(feats, p_min=float(p_min), p_max=float(p_max))
                    f.write(f"  y_base(sample): {_fmt_vec_list(y_b[0])}\n")
                    f.write(f"  y_flex_purchase(sample): {_fmt_vec_list(y_fp[0])}\n")
                    f.write(f"  y_flex_delivery(sample): {_fmt_vec_list(y_fd[0])}\n")
            except Exception as e:
                f.write(f"  [warn] Failed to log sample thresholds: {e}\n")
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





