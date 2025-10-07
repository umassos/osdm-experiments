import math
import torch
import cvxpy as cp
from functions import load_scenarios_with_flexible
from pald_implementation import (
    make_pald_base_layer,
    make_pald_flex_purchase_layer,
    make_pald_flex_delivery_layer,
    torch_objective,
    hinge_from_y_torch
)
from load_signal_trace import load_signal_trace_with_context
from paad_implementation import get_alpha
import paad_implementation as pi
from contextual_model import ThresholdPredictor
from paad_implementation import objective_function as np_objective_function
import argparse
import csv
import statistics
from datetime import datetime  # run start time + tag
from typing import List, Dict, Any
import pickle
import numpy as np
import os

try:
    import opt_sol  # offline optimal via Gurobi
    _HAS_GUROBI = True
except Exception:
    opt_sol = None
    _HAS_GUROBI = False

parser = argparse.ArgumentParser(description="Evaluate PALD-Contextual and PAAD over many instances.")
parser.add_argument("--num_instances", type=int, default=100, help="Number of instances to evaluate (default: 1)")
parser.add_argument("--trace", type=str, default="CAISO", help="Trace name (default from file config)")
parser.add_argument("--saved_model_dir", type=str, default="best_models_v1/", help="Directory with learned models (default: best_models_v1/)")
parser.add_argument("--T", type=int, default=48, help="Time horizon (default: 48)")
parser.add_argument("--gamma", type=float, default=10.0, help="Gamma switching cost parameter (default: 10.0)")
parser.add_argument("--delta", type=float, default=5.0, help="Delta switching cost parameter (default: 5.0)")
parser.add_argument("--c_delivery", type=float, default=0.2, help="Delivery cost coefficient (default: 0.2)")
parser.add_argument("--eps_delivery", type=float, default=0.05, help="Delivery cost epsilon (default: 0.05)")
parser.add_argument("--scale_factor", type=float, default=40.0, help="Scale factor for demands (default: 40.0)")
parser.add_argument("--proportion_base", type=float, default=0.5, help="Proportion of base demand (default: 0.5)")
parser.add_argument("--eta", type=float, default=0.0, help="Tracking cost parameter eta (default: 0.0, no tracking cost)")
args = parser.parse_args()

model_device = torch.device("cpu")

taus_full_t = torch.linspace(0.0, 1.0, 11, dtype=torch.float32)

# -------------------------
# Config
# -------------------------
T = args.T
S = 1.0
K = 10
gamma = args.gamma
delta = args.delta
c_delivery = args.c_delivery
eps_delivery = args.eps_delivery
scale_factor = args.scale_factor
proportion_base = args.proportion_base
model_dir = args.saved_model_dir

## CVX layers (base, flex purchase, flex delivery)
pald_base_layer = make_pald_base_layer(K)
pald_flex_purchase_layer = make_pald_flex_purchase_layer(K)
pald_flex_delivery_layer = make_pald_flex_delivery_layer(K)

def _safe_layer_call(layer, y, args, size=1.0):
    """
    Call a CvxpyLayer and catch SCS/diffcp failures. Returns x_total tensor.
    """
    try:
        (x_total,) = layer(*args)
        return x_total
    except Exception as e:
        print(f"[warning] CvxpyLayer call failed: {e}")
        print("Current y:", [float(v) for v in y.detach().cpu().reshape(-1)])
        print("Current args:", args)
        exit(1)

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


def forward_pald(price_seq, time_seq, month_seq, forecast_seq, base_seq, flex_seq, Delta_seq, model, p_min, p_max):
    # global storage state and decision both start at 0
    storage_state = torch.tensor(0.0, dtype=torch.float32)
    x_prev_global = torch.tensor(0.0)

    adjusted_S = min(sum(base_seq), 1.0)

    # adjusted_S = (1.0) * (40.0/args.scale_factor)  # storage capacity in physical units 

    # Each base driver tracks fractional progress (unit capacity); demand scales the fractional decision
    base_drivers = []  # list of dicts with keys: id, b (demand), w (fraction), prev_decision (fraction)
    # Predict threshold for this base driver
    feats = build_driver_features(
        t_idx=0, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
        storage_state=0.0,
        kind="base", b_or_f=adjusted_S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
    )
    y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
    # log y_vec_t to the console
    # print(f"[debug] Initial base y_vec: {[float(v) for v in y_vec_t.detach().cpu()]}")
    base_drivers.append({
        "id": 0,
        "b": adjusted_S,
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
                storage_state=storage_state,
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
                    storage_state=0.0,
                    kind="base", b_or_f=adjusted_S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
                )
                y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
                base_drivers.append({"id": 0, "b": adjusted_S, 
                                        "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1), 
                                        "prev_decision": torch.tensor(0.0, dtype=torch.float32).reshape(-1), "y_vec": y_vec_t})
            else:
                # Predict threshold for this base driver
                feats = build_driver_features(
                    t_idx=t, T=T, price_seq=price_seq, time_seq=time_seq, month_seq=month_seq, forecast_seq=forecast_seq,
                    storage_state=storage_state,
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
            if t == max(0, int(fd["delta"])-1):
                # need to deliver remainder
                cur_frac_z = torch.clamp(1.0 - v_prev_frac, min=0.0).reshape(-1)
                cur_phys_z = torch.mul(cur_frac_z, f_i).reshape(-1)
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
            if t == max(0, int(fd["delta"])-1):
                # need to buy remainder
                cur_frac_x = torch.clamp(1.0 - w_prev_frac, min=0.0).reshape(-1)
                cur_phys_x = torch.mul(cur_frac_x, f_i).reshape(-1)
            else:
                # x_prev_clamped = max(0.0, min(1.0 - w_eff, float(pseudo_prev_x)))

                # === forward ===
                w_hinge_t, c1_t = hinge_from_y_torch(taus_full_t, y_vec_t)

                gamma_t = torch.tensor(gamma, dtype=torch.float32).reshape(-1)
                p_t_t = torch.tensor([p_t_val], dtype=torch.float32)

                cur_frac_x = _safe_layer_call(
                    pald_flex_purchase_layer, y_vec_t, (pseudo_prev_x, w_eff, p_t_t, gamma_t, w_hinge_t, c1_t)
                )
            
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

        # Ensure purchases cover deliveries (inventory feasibility)
        # storage_state is maintained as a torch scalar throughout
        if z_t - storage_state > x_t:
            x_t = x_t + (z_t - storage_state - x_t)

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
                storage_state=adjusted_S,
                kind="base", b_or_f=adjusted_S, delta_idx=T, p_min=float(p_min), p_max=float(p_max)
            )
            y_vec_t, _, _ = model(feats, p_min=float(p_min), p_max=float(p_max))
            base_drivers.append({"id": 0, "b": adjusted_S,
                                    "w": torch.tensor(0.0, dtype=torch.float32).reshape(-1),
                                    "prev_decision": torch.tensor(0.0, dtype=torch.float32).reshape(-1), "y_vec": y_vec_t})
        # storage_state already clamped    
        
        # record sequences for torch objective
        x_hist.append(x_t)
        z_hist.append(z_t)

    # Convert sequences for torch objective
    x_torch = torch.stack(x_hist) if x_hist else torch.ones(T)
    z_torch = torch.stack(z_hist) if z_hist else torch.zeros(T)

    return x_torch.detach().numpy(), z_torch.detach().numpy(), float(storage_state.detach().item())

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

def evaluate_many(price_all, times_all, months_all, forecast_all, base_all, flex_all, Delta_all, p_min, p_max,
                   model, args, month=None):
    num_instances = len(price_all)
    print(f"Evaluating {num_instances} instances...")

    pald_costs = []
    paad_costs = []
    opt_costs = []
    pald_delivered = []
    paad_delivered = []
    opt_delivered = []

    rows: List[Dict[str, Any]] = []

    opt_recompute = True

    # check if optimal solutions are saved
    total_instances = args.num_instances
    if os.path.exists(f"eval_opt_sols/opt_costs_flex_{args.trace}_{month}_{args.num_instances}.pkl"):
        with open(f"eval_opt_sols/opt_costs_flex_{args.trace}_{month}_{args.num_instances}.pkl", "rb") as f:
            opt_costs, total_demands_saved = pickle.load(f)
        print(f"Loaded precomputed OPT costs for flexible demand from eval_opt_sols/opt_costs_flex_{args.trace}_{month}_{args.num_instances}.pkl") 
        opt_recompute = False

    for idx in range(num_instances):
        p_seq = price_all[idx]
        times_seq = times_all[idx]
        month_seq = months_all[idx]
        forecast_seq = forecast_all[idx]
        b_seq = base_all[idx]
        f_seq = flex_all[idx]
        D_seq = Delta_all[idx]

        # PALD-Fast
        pald_x, pald_z, storage_state = forward_pald(p_seq, times_seq, month_seq, forecast_seq, b_seq, f_seq, D_seq, model, p_min, p_max)
        last_price = p_seq[-1]
        pald_cost = np_objective_function(T, p_seq, gamma, delta, c_delivery, eps_delivery, pald_x, pald_z) - storage_state * last_price - gamma*storage_state
        
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
        if opt_recompute:
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
        else:
            if idx < len(opt_costs):
                oc = opt_costs[idx]
                row["opt_cost"] = oc
                delivered_opt = total_demands_saved[idx]  # assuming total demand is delivered in optimal
                row["opt_delivered"] = delivered_opt
                row["pald_over_opt"] = pald_cost / oc if oc > 0 else None
                row["paad_over_opt"] = paad_cost / oc if oc > 0 else None
            else:
                row["opt_cost"] = None
        rows.append(row)

        # if the competitive ratio is very high, print the instance details for debugging
        if "pald_over_opt" in row and row["pald_over_opt"] is not None and row["pald_over_opt"] > 2.0:
            print(f"[warning] High PALD competitive ratio {row['pald_over_opt']:.2f} on instance {idx} (month {month})")
            print(f"  Prices: {p_seq}")
            print(f"  Base demands: {b_seq}")
            print(f"  Flexible demands: {f_seq}")
            print(f"  Deadlines: {D_seq}")
            print(f"  PALD purchased: {float(sum(pald_x))}")
            print(f"  PALD delivered: {float(sum(pald_z))}")
            print(f"  PALD cost: {pald_cost}")
            print(f"  PALD ratio of purchasing to delivery: {float(sum(pald_x)) / max(float(sum(pald_z)), 1e-6):.2f}")
            print(f"  Adjusted PALD cost (storage credit): {adjusted_pald_cost}, storage left: {storage_state}")
            print(f"  OPT purchased: {sum(results['x']) if 'results' in locals() and results else 'N/A'}")
            print(f"  OPT delivered: {sum(results['z']) if 'results' in locals() and results else 'N/A'}")
            if "opt_cost" in row and row["opt_cost"] is not None:
                print(f"  OPT cost: {row['opt_cost']}, delivered: {row.get('opt_delivered', 'N/A')}")
            else:
                print("  OPT cost: N/A")

    return rows

def recover_context_features(price_all, base_all, flex_all, Delta_all, p_min, p_max, month, T):
    # Implement the logic to recover context features from the given data
    times_all = []
    months_all = []
    forecast_all = []

    # Load the full signal trace with context information
    signal, forecast_signal, datetime_index, _, _ = load_signal_trace_with_context(args.trace, month=month)

    # months_all is simply the value of month repeated once for each instance in price_all
    for _ in price_all:
        months_all.append([month for _ in range(T)])
    
    # to recover times_all and forecast_all, we need to load and recover the datetime indexes that correspond to each price_all instance
    for idx in range(len(price_all)):
        p_seq = price_all[idx]
        # find the start time of this instance by matching the first price in p_seq to the signal trace
        start_price = p_seq[0]
        start_time = 0
        # find all indexes in signal where the price matches start_price (within a small tolerance)
        tolerance = 1e-3
        matching_indexes = signal[(signal >= start_price - tolerance) & (signal <= start_price + tolerance)].index
        if len(matching_indexes) == 0:
            raise ValueError(f"Could not find matching start price {start_price} in signal trace.")
        if len(matching_indexes) > 1:
            # check to see which of these matching indexes has the full sequence p_seq following it
            found = False
            for mi in matching_indexes:
                # get the slice of signal starting at mi and of length T
                mi_loc = datetime_index.get_loc(mi)
                slice_signal = signal[mi_loc:mi_loc + len(p_seq)]
                if len(slice_signal) == len(p_seq) and all(abs(slice_signal.values - p_seq) < tolerance):
                    start_time = mi
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not find full matching sequence starting with price {start_price} in signal trace.")
        else:
            start_time = matching_indexes[0]

        # get the array index of start_index in datetime_seq
        start_idx = datetime_index.get_loc(start_time)
        
        # datetime sequence is the range from start_index to start_index + T
        datetime_seq = datetime_index[start_idx:start_idx + T]
        if len(datetime_seq) < T:
            raise ValueError(f"Not enough data in signal trace starting from {start_time} for full sequence.")
        times_all.append(datetime_seq)

        # forecast sequence is the corresponding values from forecast_signal
        forecast_seq = forecast_signal[start_idx:start_idx + T]
        if len(forecast_seq) < T:
            raise ValueError(f"Not enough data in forecast signal trace starting from {start_time} for full sequence.")
        forecast_all.append(forecast_seq)


    return times_all, months_all, forecast_all

def main():
    max_month = 12

    print(f"Evaluating {args.num_instances} instances (trace={args.trace})...")
    month_data = []
    threshold_data = []
    for month in range(1, max_month + 1):
        price_all, base_all, flex_all, Delta_all, p_min, p_max = load_scenarios_with_flexible(
            args.num_instances, T, args.trace, month=month, eval=True, scale_factor=scale_factor, proportion_base=proportion_base
        )
        print(f"Month {month}: Loaded {len(price_all)} instances with p_min={p_min}, p_max={p_max}")

        tracking_target_all = [[0.0 for _ in seq] for seq in base_all]
        if scale_factor != 40.0:
            # rescale demands by the new scale factor (e.g., if scale factor = 80, divide all demands by 2)
            divisor = scale_factor / 40.0
            base_all = [[b / divisor for b in seq] for seq in base_all]
            flex_all = [[f / divisor for f in seq] for seq in flex_all]

        # recover context about this particular set of instances
        times_all, months_all, forecast_all = recover_context_features(price_all, base_all, flex_all, Delta_all, p_min, p_max, month, T)

        month_data.append((price_all, times_all, months_all, forecast_all, base_all, flex_all, Delta_all, p_min, p_max))
    
    # check if file exists
    try:
        # look for file in best_models directory
        directory = args.saved_model_dir
        # search for a file in this directory that contains args.trace in the file name
        import os
        files = os.listdir(directory)
        matching_files = [f for f in files if args.trace in f and f.endswith('.pt')]
        if not matching_files:
            print(f"No model file found in {directory} matching trace {args.trace}")
            return
        # if multiple matching files, take the first one (should only be one)
        PATH = os.path.join(directory, matching_files[0])
        print(f"Found model file: {PATH}")
        # f should contain the model (as a torch.save file) -- we should now load it
        model = ThresholdPredictor(input_dim=11, K=K+1, hidden_dims=(64, 64), beta=100, p_min=float(p_min), p_max=float(p_max), use_robust_projection=True)
        model.load_state_dict(torch.load(PATH, weights_only=False))
        model.eval()
        print(f"Loaded model from {PATH}")
    except FileNotFoundError:
        print(f"Unable to load model file from {args.saved_model_dir}")
        return

    rows = []
    for month in range(1, max_month + 1):
        print(f"\n--- Evaluating for Month {month} ---")
        price_all, times_all, months_all, forecast_all, base_all, flex_all, Delta_all, p_min, p_max = month_data[month - 1]
        rows_month = evaluate_many(price_all, times_all, months_all, forecast_all, base_all, flex_all, Delta_all, p_min, p_max, model, args, month=month)
        rows.extend(rows_month)
    
    # Aggregates
    print("\n=== Aggregate Results ===")
    def print_summary(label, vals):
        if not vals:
            print(f"{label}: (none)")
            return
        s = summarize(vals)
        print(f"{label}:    mean={s['mean']:.4f}    median={s['median']:.4f}    p95={s['p95']:.4f}    min={s['min']:.4f}")

    ratios_pald = [r["pald_over_opt"] for r in rows if r.get("pald_over_opt")]
    ratios_paad = [r["paad_over_opt"] for r in rows if r.get("paad_over_opt")]
    # truncate ratios to 1.0
    ratios_pald = [max(1.0, r) for r in ratios_pald]
    ratios_paad = [max(1.0, r) for r in ratios_paad]
    print_summary("PALD/OPT", ratios_pald)
    print_summary("PAAD/OPT", ratios_paad)

    # Save detailed results to a pickle file
    # extract parameters for file name out of args.saved_model_dir
    prefix = args.saved_model_dir.split('/')[0]
    output_file = f'eval_results/{prefix}_{args.trace}_T{args.T}_gamma{args.gamma}_delta{args.delta}_c{args.c_delivery}_eps{args.eps_delivery}_prop{proportion_base}_scale{scale_factor}.pkl'
    os.makedirs('eval_results', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(rows, f)
    print(f"Saved detailed results to {output_file}")

    # plot CDF of the ratios
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        def plot_cdf(data, label, color):
            sorted_data = np.sort(data)
            yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
            plt.plot(sorted_data, yvals, label=label, color=color)

        plt.figure(figsize=(4, 3), dpi=300)
        if ratios_pald:
            plot_cdf(ratios_pald, 'PALD', 'blue')
        if ratios_paad:
            plot_cdf(ratios_paad, 'PAAD', 'orange')
        plt.xlabel('Comp. Ratio')
        plt.ylabel('Cumulative Probability')
        # legend at the bottom in two columns
        plt.legend()
        plt.grid(True)
        plt.xlim(1, 4)
        plt.ylim(0, 1)
        plt.savefig(f'comp_ratio_cdf_{args.trace}.png')
        plt.close()
        print(f"Saved CDF plot to comp_ratio_cdf_{args.trace}.png")
    except ImportError:
        print("matplotlib not installed, skipping CDF plot.")

if __name__ == "__main__":
    main()

