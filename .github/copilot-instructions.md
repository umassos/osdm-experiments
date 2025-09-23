# Copilot instructions for this repo

Purpose: This repo prototypes Online Storage and Demand Management (OSDM) experiments. Two main tracks:
- Offline optimal benchmark via Gurobi (quadratic objective with absolute-value linearization).
- Online PAAD algorithm (driver-based policy) with small per-step convex subproblems; plus a toy PALD training loop using CVXPyLayers + PyTorch.

Key modules and roles
- `initial_exp.py`: Entry experiment. Builds a 24h horizon, loads a price trace, defines demands, runs `opt_sol.optimal_solution(...)` (offline) then `paad_implementation.paad_algorithm(...)` (online), and prints metrics.
- `paad_implementation.py`: Core online algorithm and objective.
  - `objective_function(...)`: Computes OSDM objective for given sequences.
  - `BaseDriver` and `FlexibleDriver`: Encapsulate per-demand dynamics, pseudo-decision accounting, and per-step convex minimizations (purchase/delivery) solved with CVXPY (solver Clarabel).
  - `get_alpha(...)`: Competitive ratio via `scipy.special.lambertw` used in thresholds.
  - `paad_algorithm(...)`: Orchestrates drivers over time; creates new drivers when base (`b_t`) or flexible (`f_t, Δ_f,t`) demand arrives; aggregates `x_t, z_t`, maintains storage `s_t`.
- `opt_sol.py`: Offline optimal with Gurobi. Builds variables `{x_t, z_t, s_t}`, absolute-value auxiliaries, state dynamics, cumulative flexible-deadline constraint, and quadratic term `-p_t c s_{t-1} z_t`. Requires Gurobi license; otherwise skip this path.
- `functions.py`: Scenario builder for training (toy). `load_scenarios(...)` slices historical traces into sequences and returns `(price_scenarios, demand_scenarios, p_min, p_max)`.
- `load_signal_trace.py`: Loads CSV traces in `signal_traces/`, selects column by type, returns `(pd.Series signal, DatetimeIndex, p_min, p_max)`. Filenames are matched by substring against curated path lists.
- `pald_implementation.py`: Defines `make_pald_base_layer(K, gamma)` as a CVXPyLayer (one-step base driver with cost `p_t x + γ|x−x_prev| + γ|x| − y x`).
- `train_pald.py`: Toy training loop that learns a piecewise-linear price transform `y ∈ R^K` by backprop through `CvxpyLayer` using Adam.

Data/model flow (PAAD)
1) Price `p_t` comes from a trace; base demand `b_t` and flexible demand/deadline `(f_t, Δ_f,t)` are exogenous.
2) At each `t`, drivers compute pseudo-decisions from previous global decisions; each driver solves a small CVXPy problem to pick its `x` (and `z` for flexible), then the system aggregates `x_t = Σ x`, `z_t = b_t + Σ z` and updates storage `s_t = s_{t−1} + x_t − z_t`.
3) If storage empties, base drivers reset (see `paad_algorithm`). `objective_function` is used to evaluate the final policy cost.

Conventions and gotchas
- Time horizon `T` typically 24; units scaled to `S = 1`. See `initial_exp.py`.
- `paad_implementation.paad_algorithm` currently hardcodes `p_min, p_max` (CAISO). When generalizing, prefer computing these from the active `p` sequence.
- CVXPy problems use solver `CLARABEL` (convex). Keep expressions DCP-compliant if modifying the driver cost/threshold forms. A few early-return guards avoid problem setup when a condition is trivially optimal.
- `load_signal_trace(...)` matches filenames by substring; ensure new files are added to the curated lists and referenced via a substring (e.g., `"CAL"` matches `"US-CAL-CISO.csv"`).
- Mixing frameworks: PAAD path uses NumPy/CVXPY; PALD training uses PyTorch/CVXPYLayers. Keep tensors vs floats consistent per path.

Dependencies
- Python 3.10+; required: `numpy`, `pandas`, `scipy`, `cvxpy`, `clarabel`, `cvxpylayers`, `torch`.
- Optional (offline solver & plotting): `gurobipy` (license), `matplotlib`, `seaborn`, `tqdm`.

Quickstart (zsh)
```zsh
# minimal runtime deps
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scipy cvxpy clarabel cvxpylayers torch

# optional (offline optimal via Gurobi)
pip install gurobipy matplotlib seaborn tqdm

# run the example experiment
python initial_exp.py

# run the PALD training toy example
python train_pald.py
```

Extending/using the code
- Add new traces: drop CSV under `signal_traces/` and register the path in `load_signal_trace.py`.
- New online policies: follow the `BaseDriver`/`FlexibleDriver` pattern—maintain `prev_*`, `pseudo_*`, and cumulative progress (`w`, `v`); expose a per-step CVXPy minimization and call it from the main loop.
- Integrate a new solver: use CVXPy-compatible convex forms; set `prob.solve(solver=cp.CLARABEL)` or another installed solver.

Examples
- Call PAAD directly:
```python
from paad_implementation import paad_algorithm
results = paad_algorithm(T, p, gamma, delta, c, eps, S, b, f, Delta_f)
print(results['obj_val'], sum(results['x']), sum(results['z']))
```

Notes for AI agents
- Preserve shapes: sequences are Python lists of length `T`; driver internals use floats/NumPy. CVXPyLayer in `train_pald.py` expects tensors (`torch`) with compatible scalar inputs.
- Be cautious editing threshold/antiderivative forms: they interact with convexity and solver selection.