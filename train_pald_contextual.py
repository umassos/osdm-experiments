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
    compute_segment_caps,
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

solver_options = {
    # SCS-like settings often work best for differentiable layers
    "eps": 1e-5,
    "max_iters": 2000,
    "verbose": False,
}

parser = argparse.ArgumentParser(description="Train PALD with flexible demand and deadlines.")
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4 for debugging)')
parser.add_argument('--num_batches', type=int, default=1, help='Number of batches per epoch (default: 1)')
parser.add_argument('--use_cost_loss', action='store_true', help='Use total cost loss instead of competitive-ratio loss')
parser.add_argument('--trace', type=str, default="CAISO", help='Trace name to use (default: CAISO)')
parser.add_argument('--month', type=int, default=99, help='Month to filter for in trace (default: 1, 99 for all)')
parser.add_argument('--warmup_epochs', type=int, default=50, help='Supervised warm-up epochs to aligning y0 to OPT/quantile targets')
parser.add_argument('--warmup_lambda', type=float, default=5.0, help='Weight of warm-up y0 loss during warm-up phase')
parser.add_argument('--y0_margin', type=float, default=10.0, help='Margin to add to OPT avg price for base/flex purchase y0 target')
parser.add_argument('--post_warmup_epochs', type=int, default=50, help='Number of epochs to decay anchor after warm-up')
parser.add_argument('--post_warmup_lambda', type=float, default=1.0, help='Initial weight of post-warmup anchor (decays to 0)')
parser.add_argument('--freeze_trunk_epochs', type=int, default=20, help='Freeze trunk for these epochs after warm-up to avoid collapse')
parser.add_argument('--topup_penalty_lambda', type=float, default=20.0, help='Weight for penalizing reliance on forced top-ups')
args = parser.parse_args()

K = 10           # number of segments in piecewise linear approximation for psi
gamma = 10.0     # switching cost parameter for x
delta = 5.0     # switching cost parameter for z (used in analytical threshold)
S = 1.0          # maximum inventory capacity
T = 48          # 12 hours in 15-minute intervals
c_delivery = 0.2
eps_delivery = 0.05
epochs = 500
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

## Contextual thresholds model (MonotoneHead with softplus cumulative form)

# Instantiate contextual model
feature_dim = 11  # handcrafted features length (see build_driver_features)
model = ThresholdPredictor(input_dim=feature_dim, K=K, hidden_dims=(128, 128))
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
pald_base_layer = make_pald_base_layer(K, gamma, ridge=False)
pald_flex_purchase_layer = make_pald_flex_purchase_layer(K, gamma, ridge=False)
pald_flex_delivery_layer = make_pald_flex_delivery_layer(K, delta, c_delivery, eps_delivery, ridge=False)

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


## Differentiable PALD objective in torch (matches paad_implementation.objective_function)
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
    # s_{t-1} sequence for discharge term
    s_prev_seq = torch.cat([s_torch[:-1]])
    discharge_cost = (p_seq * (c * z_seq + eps * z_seq - c * s_prev_seq * z_seq)).sum()
    return cost_purchasing + switching_cost_x + switching_cost_z + discharge_cost


def _safe_layer_call(layer, args, size=1.0):
    """Call a CvxpyLayer and unwrap its single-output tensor."""
    (val,) = layer(*args, solver_args=solver_options)
    return val


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
        # Enable/disable parameter groups depending on warm-up
        warmup_active = epoch < int(args.warmup_epochs)
        if warmup_active:
            # Freeze everything except the top gate biases/weights to shape y0
            for name, p in model.named_parameters():
                if any(k in name for k in ["_head_top.weight", "_head_top.bias"]):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)
        else:
            # Immediately after warm-up, freeze trunk for a few epochs to stabilize heads
            post_warm = epoch - int(args.warmup_epochs)
            for name, p in model.named_parameters():
                if post_warm >= 0 and post_warm < int(args.freeze_trunk_epochs):
                    if any(k in name for k in ["trunk."]):
                        p.requires_grad_(False)
                    else:
                        p.requires_grad_(True)
                else:
                    p.requires_grad_(True)
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
                storage_state = 0.0
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
                base_drivers.append({"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0, "y_vec": y_vec_t})
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
                        flex_drivers.append({"id": 2 * t + 1, "f": f_arrival, "delta": dlt, "w": 0.0, "v": 0.0, "prev_x": 0.0, "prev_z": 0.0, "y_vec_purchase": y_vec_p, "y_vec_delivery": y_vec_d})

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
                            base_drivers.append({"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0, "y_vec": y_vec_t})
                        else:
                            # Predict threshold for this base driver
                            feats = build_driver_features(
                                t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                                storage_state=S,
                                kind="base", b_or_f=b_t_val, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                            )
                            y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                            base_drivers.append({"id": 2 * t + 2, "b": b_t_val, "w": 0.0, "prev_decision": 0.0, "y_vec": y_vec_t})

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

                    # compute the cumulative upper bound on the buying decision at the current time step:
                    # this buy cap is (S - storage_state) + possible z_t
                    
                    # first determine the flex deliveries

                    # Base delivery equals current base demand arrival
                    z_components = [torch.tensor(b_t_val, dtype=torch.float32)]

                    # Flexible drivers: delivery decisions
                    for fd in flex_drivers:
                        f_i = fd["f"]
                        prev_frac_z = fd["prev_z"]
                        y_vec_d = fd["y_vec_delivery"]
                        # share positive excess proportional to previous physical contribution
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
                            y_vec_d = fd["y_vec_delivery"]
                            z_prev_frac_t = torch.tensor(z_prev_clamped, dtype=torch.float32)
                            v_prev_frac_t = torch.tensor(v_eff, dtype=torch.float32)
                            p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                            s_prev_t = torch.tensor(float(max(0.0, storage_state)), dtype=torch.float32)
                            caps_t = torch.tensor(caps_list, dtype=torch.float32)

                            # Precompute coeff = p_t * (c+eps) - p_t * c * s_prev  (scalar)
                            coeff_t = torch.tensor(
                                p_t_val * ((c_delivery + eps_delivery) - c_delivery * float(max(0.0, storage_state))),
                                dtype=torch.float32,
                            )

                            cur_frac_z = _safe_layer_call(
                                pald_flex_delivery_layer, (z_prev_frac_t, v_prev_frac_t, coeff_t, y_vec_d, caps_t), size=(1.0 - v_eff)
                            )
                        cur_phys_z = torch.mul(cur_frac_z, f_i)
                        z_components.append(cur_phys_z)
                        fd["prev_z"] = float(cur_frac_z.detach())
                        fd["v"] = float(min(1.0, fd["v"] + fd["prev_z"]))

                    z_t = torch.stack(z_components).sum()
                    storage_torch = torch.tensor(storage_state, dtype=torch.float32)

                    # now that we have the delivery z_t, we can compute the buy cap
                    buy_cap = float(S) - storage_state + float(z_t.detach())
                    # we will decrement from this buy_cap as we allocate to drivers below

                    # Determine per-driver decisions (fractional)
                    decisions = []  # list of tensors in physical units
                    for drv in base_drivers:
                        if buy_cap <= 1e-9:
                            # no more buying possible, force zero decision
                            cur_phys_decision = torch.tensor(0.0, dtype=torch.float32)
                            decisions.append(cur_phys_decision)
                            drv["prev_decision"] = float(cur_phys_decision.detach())
                            drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))
                            continue

                        b_i = drv["b"]
                        prev_frac = drv["prev_decision"]
                        y_vec_t = drv["y_vec"]

                        # share positive excess proportional to previous physical contribution
                        denom = prev_purchasing_total if prev_purchasing_total > 0 else 1.0
                        share = (prev_frac * b_i) / denom if prev_purchasing_total > 0 else 0.0
                        pseudo_prev_frac = prev_frac + max(0.0, purchasing_excess) * share / max(b_i, 1e-8)

                        # Compute per-segment caps for current cumulative fraction w
                        w_prev_frac = float(drv["w"])
                        # Clamp w into [0, 1 - eps] to avoid issues with the solver
                        w_eff = max(0.0, min(1.0 - 1e-9, w_prev_frac))
                        caps_list = compute_segment_caps(w_eff, K)
                        if (1.0 - w_eff) <= 1e-9 or sum(caps_list) <= 1e-12:
                            cur_frac_decision = torch.tensor(0.0, dtype=torch.float32)
                            cur_phys_decision = torch.mul(cur_frac_decision, b_i)
                            decisions.append(cur_phys_decision)
                            drv["prev_decision"] = float(cur_frac_decision.detach())
                            drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))
                            continue

                        x_prev_frac_t = torch.tensor(float(pseudo_prev_frac), dtype=torch.float32)
                        w_prev_frac_t = torch.tensor(w_eff, dtype=torch.float32)
                        p_t_t = torch.tensor(p_t_val, dtype=torch.float32)
                        caps_t = torch.tensor(caps_list, dtype=torch.float32)

                        cur_frac_decision = _safe_layer_call(
                            pald_base_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y_vec_t, caps_t), size=(1.0 - w_eff)
                        )
                        # diagnostics: base activation
                        epoch_base_calls += 1
                        if float(cur_frac_decision.detach().item()) > 1e-9:
                            epoch_base_active += 1

                        # Convert to physical units by scaling with demand of this driver
                        cur_phys_decision = torch.mul(cur_frac_decision, b_i)

                        # check if this decision exceeds the remaining buy cap
                        if float(cur_phys_decision.detach()) > buy_cap + 1e-5:
                            # take the remaining buy cap instead
                            cur_phys_decision = torch.tensor(buy_cap, dtype=torch.float32)
                            # and set the fractional decision accordingly
                            if b_i > 1e-8:
                                cur_frac_decision = cur_phys_decision / b_i
                            else:
                                cur_frac_decision = torch.tensor(0.0, dtype=torch.float32)
                            # after this, the buy cap is zero
                            buy_cap = 0.0

                        decisions.append(cur_phys_decision)
                        buy_cap -= float(cur_phys_decision.detach())

                        # Update driver internal state (detach to avoid history growth)
                        drv["prev_decision"] = float(cur_frac_decision.detach())
                        drv["w"] = float(min(1.0, drv["w"] + drv["prev_decision"]))

                    # Flexible drivers: purchasing decisions
                    for fd in flex_drivers:
                        if buy_cap <= 1e-9:
                            # no more buying possible, force zero decision
                            cur_phys_x = torch.tensor(0.0, dtype=torch.float32)
                            decisions.append(cur_phys_x)
                            fd["prev_x"] = float(cur_phys_x.detach())
                            fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))
                            continue

                        f_i = fd["f"]
                        prev_frac_x = fd["prev_x"]
                        y_vec_t = fd["y_vec_purchase"]

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
                            caps_t = torch.tensor(caps_list, dtype=torch.float32)

                            cur_frac_x = _safe_layer_call(
                                pald_flex_purchase_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y_vec_t, caps_t), size=(1.0 - w_eff)
                            )
                        # diagnostics: flex purchase activation
                        epoch_flex_calls += 1
                        if float(cur_frac_x.detach().item()) > 1e-9:
                            epoch_flex_active += 1
                        
                        cur_phys_x = torch.mul(cur_frac_x, f_i)

                        # check if this decision exceeds the remaining buy cap
                        if float(cur_phys_x.detach()) > buy_cap + 1e-5:
                            # take the remaining buy cap instead
                            cur_phys_x = torch.tensor(buy_cap, dtype=torch.float32)
                            # and set the fractional decision accordingly
                            if f_i > 1e-8:
                                cur_frac_x = cur_phys_x / f_i
                            else:
                                cur_frac_x = torch.tensor(0.0, dtype=torch.float32)
                            # after this, the buy cap is zero
                            buy_cap = 0.0
                        
                        decisions.append(cur_phys_x)
                        buy_cap -= float(cur_phys_x.detach())

                        fd["prev_x"] = float(cur_frac_x.detach())
                        fd["w"] = float(min(1.0, fd["w"] + fd["prev_x"]))

                    # Aggregate physical purchases this step
                    x_t = torch.stack(decisions).sum() if decisions else torch.tensor(0.0)
                    # Diagnostics: forced top-up outside CVX layers
                    x_pre_val = float(x_t.detach().item())
                    
                    # Ensure purchases cover deliveries (inventory feasibility)
                    required = torch.maximum(z_t - storage_torch, torch.tensor(0.0))
                    forced_extra = torch.maximum(required - torch.tensor(x_pre_val, dtype=torch.float32), torch.tensor(0.0))
                    if forced_extra > 1e-9:
                        epoch_forced_topup_sum += forced_extra.item()
                        epoch_forced_topup_events += 1
                    epoch_xpre_count += 1
                    if x_pre_val <= 1e-9:
                        epoch_xpre_zero += 1
                    x_t = torch.maximum(x_t, z_t - torch.tensor(storage_state, dtype=x_t.dtype))

                    # diagnostics -- check if the currect decision will ``overfill the storage''
                    if float(storage_state + float(x_t.detach()) - float(z_t.detach())) > S + 1e-3:
                        print(f"[warning] t={t} overfill: storage {storage_state:.3f} + x {float(x_t.detach()):.3f} - z {float(z_t.detach()):.3f} > S={S}")
                    
                    storage_state_next = float(storage_state + float(x_t.detach()) - float(z_t.detach()))
                    x_prev_global = x_t.detach()

                    # if the storage will be empty and it was previously non-empty, we can refresh the base drivers
                    if storage_state_next <= 1e-9 and storage_state > 1e-9:
                        # print(f"[info] t={t} storage emptied, refreshing base drivers")
                        base_drivers = []  # reset previous base drivers
                        feats = build_driver_features(
                            t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                            storage_state=S,
                            kind="base", b_or_f=S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                        )
                        y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                        base_drivers.append({"id": 0, "b": S, "w": 0.0, "prev_decision": 0.0, "y_vec": y_vec_t})

                    storage_state = max(0.0, min(S, storage_state_next))    
                    
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
    print("[train] Caught KeyboardInterrupt. Skipping remaining training and running evaluation...")

# Evaluate PALD, PAAD, and OPT on the first instance and plot time series
# Forward PALD rollout with learned thresholds (for visualization only)
def forward_pald(price_seq, time_seq, month_seq, forecast_seq, base_seq, flex_seq, Delta_seq):
    x_list, z_list, s_list = [], [], []
    storage_state = 0.0
    x_prev_global = torch.tensor(0.0)

    # thresholds predicted per-driver via model

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
                    y_pred, _, _ = model(build_driver_features(
                        t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                        storage_state=storage_state,
                        kind="base", b_or_f=b_i, delta_idx=t, p_min=float(p_min), p_max=float(p_max)
                    ), p_min=float(p_min), p_max=float(p_max))
                    cur_frac_decision = _safe_layer_call(
                        pald_base_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y_pred, caps_t), size=(1.0 - w_eff)
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
                    _, y_fp_pred, _ = model(build_driver_features(
                        t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                        storage_state=storage_state,
                        kind="flex", b_or_f=f_i, delta_idx=int(fd["delta"]), p_min=float(p_min), p_max=float(p_max)
                    ), p_min=float(p_min), p_max=float(p_max))
                    cur_frac_x = _safe_layer_call(
                        pald_flex_purchase_layer, (x_prev_frac_t, w_prev_frac_t, p_t_t, y_fp_pred, caps_t), size=(1.0 - w_eff)
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
                    _, _, y_fd_pred = model(build_driver_features(
                        t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                        storage_state=storage_state,
                        kind="flex", b_or_f=f_i, delta_idx=int(fd["delta"]), p_min=float(p_min), p_max=float(p_max)
                    ), p_min=float(p_min), p_max=float(p_max))
                    cur_frac_z = _safe_layer_call(
                        pald_flex_delivery_layer, (z_prev_frac_t, v_prev_frac_t, coeff_t, y_fd_pred, caps_t), size=(1.0 - v_eff)
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
    t0 = times_all[0]
    m0 = months_all[0]
    fc0 = forecast_all[0]
    b0 = base_all[0]
    f0 = flex_all[0]
    Delta0 = Delta_all[0]

    # PALD forward pass with learned y
    pald_x, pald_z, pald_s = forward_pald(p0, t0, m0, fc0, b0, f0, Delta0)

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





