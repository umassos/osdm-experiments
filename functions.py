from load_signal_trace import load_signal_trace, load_signal_trace_with_context
import trace_loader
import numpy as np
import pickle
import opt_sol as opt_sol

ctx = trace_loader.prepare_exploded_from_csv("demand_traces/batch_task.csv", bucket_minutes=15)

def load_scenarios(num_scenarios, T, trace_name, proportion_base=0.5):
    """
    Load scenarios from pre-saved traces.
    
    Args:
        num_scenarios: Number of scenarios to load or generate
        T: Time horizon
        trace_name: Name of the trace to load
        demand_range: (min_demand, max_demand)

    Returns:
        price_scenarios, demand_scenarios
    """
    price_scenarios = []
    demand_scenarios = []

    # choose a price signal from the available traces
    filename = trace_name
    signal, datetimes, p_min, p_max = load_signal_trace(filename)

    # if we don't already have scenarios pickled for this configuration, generate them
    # first check for a pickled file in demand_traces
    pickle_filename = f"demand_traces/{trace_name}_num{num_scenarios}_deadline{T}.pkl"
    try:
        with open(pickle_filename, "rb") as f:
            price_scenarios, demand_scenarios = pickle.load(f)
        print(f"Loaded {num_scenarios} scenarios from {pickle_filename}.")
        return price_scenarios, demand_scenarios, p_min, p_max
    except FileNotFoundError:
        print(f"No pickled scenarios found at {pickle_filename}, generating new scenarios.")
        for _ in range(num_scenarios):
            # randomly choose an index from datetimes, and make sure there are at least T slots including/after that index
            index = np.random.randint(0, len(datetimes) - T)
            dtSequence = datetimes[index:index+T]

            # Get the signal trace for the sequence
            p = signal[dtSequence].tolist()

            # randomly choose an available index from the ctx to get demand traces
            bucket_min, bucket_max = int(ctx["bucket_stats"]["bucket"].min()), int(ctx["bucket_stats"]["bucket"].max())
            # choose a random start bucket index so that we have at least T buckets available
            start_bucket = np.random.randint(bucket_min, bucket_max - T)
            idxs = list(range(start_bucket, start_bucket + T))

            # get the base and flexible demand series for these indexes, scaled down by a factor of 40.0
            base_scaled, flex_scaled, deltas, details = trace_loader.compute_base_flexible_series(
                ctx, day_bucket_indexes=idxs, T=T, seed=123, scale_divisor=40.0, proportion_base=proportion_base
            )

            price_scenarios.append(p)
            demand_scenarios.append(base_scaled)

        return price_scenarios, demand_scenarios, p_min, p_max

def load_scenarios_with_flexible(num_scenarios, T, trace_name, proportion_base=0.5, month=1, saved=True, eval=False, scale_factor=40):
    """
    Load scenarios with base and flexible demand plus deadlines.

    Returns:
      price_scenarios, base_scenarios, flex_scenarios, Delta_f_scenarios, p_min, p_max
    """
    price_scenarios = []
    base_scenarios = []
    flex_scenarios = []
    Delta_scenarios = []

    signal, datetimes, p_min, p_max = load_signal_trace(trace_name, month=month)

    # if we don't already have scenarios pickled for this configuration, generate them
    # first check for a pickled file in demand_traces
    if saved:
        pickle_filename = f"demand_traces/{trace_name}_month{month}_num{num_scenarios}_deadline{T}_prop{proportion_base}_scale{scale_factor}.pkl"
        if eval:
            pickle_filename = f"eval_traces/{trace_name}_month{month}_num{num_scenarios}_deadline{T}_prop{proportion_base}_scale{scale_factor}.pkl"
        try:
            with open(pickle_filename, "rb") as f:
                price_scenarios, base_scenarios, flex_scenarios, Delta_scenarios = pickle.load(f)
            print(f"Loaded {num_scenarios} scenarios from {pickle_filename}.")
            return price_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max
        except FileNotFoundError:
            print(f"No pickled scenarios found at {pickle_filename}, generating new scenarios.")

    
    while len(price_scenarios) < num_scenarios:
        # randomly choose an index from datetimes, and make sure there are at least T slots including/after that index
        index = np.random.randint(0, len(datetimes) - T)
        dtSequence = datetimes[index:index+T]

        # Get the signal trace for the sequence
        p = signal[dtSequence].tolist()

        # randomly choose an available index from the ctx to get demand traces
        bucket_min, bucket_max = int(ctx["bucket_stats"]["bucket"].min()), int(ctx["bucket_stats"]["bucket"].max())
        # choose a random start bucket index so that we have at least T buckets available
        start_bucket = np.random.randint(bucket_min, bucket_max - T)
        idxs = list(range(start_bucket, start_bucket + T))

        ##################################### alibaba demand sequence generation
        # get the base and flexible demand series for these indexes, scaled down by a factor of 40.0
        base_scaled, flex_scaled, deltas, details = trace_loader.compute_base_flexible_series(
            ctx, day_bucket_indexes=idxs, T=T, seed=123, scale_divisor=scale_factor, proportion_base=proportion_base
        )

        ##################################### simple hardcoded demand sequence
        # # if T = 48, this is 0.5 base demand arriving at times 23 and 47
        # b = [0.0 for _ in range(T)]
        # b[23] = 0.5
        # b[47] = 0.5
        # base_scaled = b

        # # flexible demand of 0.4 arriving at times 0 and 23, with deadlines 23 and 47
        # f = [0.0 for _ in range(T)]
        # f[0] = 0.4
        # f[23] = 0.4
        # flex_scaled = f
        # deltas = [0 for _ in range(T)]
        # deltas[0] = 23
        # deltas[23] = 47

        # make sure that solving for the optimal solution is feasible
        s_0 = 0.0
        x_0 = 0.0
        z_0 = 0.0
        gamma = 10 # switching cost parameter for x
        delta = 5 # switching cost parameter for z
        c_delivery = 0.2 # parameter for the discharge cost function
        eps_delivery = 0.05 # parameter for the discharge cost function (eps_delivery < c_delivery)
        S = 1 # maximum capacity of the inventory (everything scaled to unit sizes)
        try:
            status, results = opt_sol.optimal_solution(
                T, p, gamma, delta, c_delivery, eps_delivery, S, base_scaled, flex_scaled, deltas, s_0, x_0, z_0, solver_timeout=60.0
            )
            if status != "Optimal":
                print("Generated scenario is not feasible for OPT, skipping this scenario.")
                continue
        except:
            print("Error occurred while solving for OPT, skipping this scenario.")
            continue

        price_scenarios.append(p)
        base_scenarios.append(base_scaled)
        flex_scenarios.append(flex_scaled)
        Delta_scenarios.append(deltas)


    if saved:
        # save the generated scenarios to a pickle file for future use
        with open(pickle_filename, "wb") as f:
            pickle.dump((price_scenarios, base_scenarios, flex_scenarios, Delta_scenarios), f)
        print(f"Saved generated scenarios to {pickle_filename}.")
        
    return price_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max

def load_scenarios_with_flexible_context(num_scenarios, T, trace_name, proportion_base=0.5, month=1, scale_divisor=40, saved=True):
    """
    Load scenarios with base and flexible demand plus deadlines.

    Returns:
      price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_f_scenarios, p_min, p_max
    """
    price_scenarios = []
    forecast_scenarios = []
    time_scenarios = []
    month_scenarios = []
    base_scenarios = []
    flex_scenarios = []
    Delta_scenarios = []

    signal, forecasts, datetimes, p_min, p_max = load_signal_trace_with_context(trace_name, month=month)

    # if we don't already have scenarios pickled for this configuration, generate them
    # first check for a pickled file in demand_traces
    if saved:
        pickle_filename = f"demand_traces/{trace_name}_month{month}_num{num_scenarios}_deadline{T}_prop{proportion_base}_CONTEXT.pkl"
        try:
            with open(pickle_filename, "rb") as f:
                price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios = pickle.load(f)
            print(f"Loaded {num_scenarios} scenarios from {pickle_filename}.")
            return price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max
        except FileNotFoundError:
            print(f"No pickled scenarios found at {pickle_filename}, generating new scenarios.")

    
    while len(price_scenarios) < num_scenarios:
        # randomly choose an index from datetimes, and make sure there are at least T slots including/after that index
        index = np.random.randint(0, len(datetimes) - T)
        dtSequence = datetimes[index:index+T]

        # Get the signal trace for the sequence
        p = signal[dtSequence].tolist()
        # get the forecast trace for the sequence
        forecast_p = forecasts[dtSequence].tolist()

        # using the dt sequence, extract the time (hour:min) and month as context features
        times = [dt.time() for dt in dtSequence]
        months = [dt.month for dt in dtSequence]

        # randomly choose an available index from the ctx to get demand traces
        bucket_min, bucket_max = int(ctx["bucket_stats"]["bucket"].min()), int(ctx["bucket_stats"]["bucket"].max())
        # choose a random start bucket index so that we have at least T buckets available
        start_bucket = np.random.randint(bucket_min, bucket_max - T)
        idxs = list(range(start_bucket, start_bucket + T))

        ##################################### alibaba demand sequence generation
        # get the base and flexible demand series for these indexes, scaled down by a factor of scale_divisor
        base_scaled, flex_scaled, deltas, details = trace_loader.compute_base_flexible_series(
            ctx, day_bucket_indexes=idxs, T=T, seed=123, scale_divisor=scale_divisor, proportion_base=proportion_base
        )

        # make sure that solving for the optimal solution is feasible
        s_0 = 0.0
        x_0 = 0.0
        z_0 = 0.0
        gamma = 10 # switching cost parameter for x
        delta = 5 # switching cost parameter for z
        c_delivery = 0.2 # parameter for the discharge cost function
        eps_delivery = 0.05 # parameter for the discharge cost function (eps_delivery < c_delivery)
        S = 1 # maximum capacity of the inventory (everything scaled to unit sizes)
        try:
            status, results = opt_sol.optimal_solution(
                T, p, gamma, delta, c_delivery, eps_delivery, S, base_scaled, flex_scaled, deltas, s_0, x_0, z_0, solver_timeout=60.0
            )
            if status != "Optimal":
                print("Generated scenario is not feasible for OPT, skipping this scenario.")
                continue
        except:
            print("Error occurred while solving for OPT, skipping this scenario.")
            continue

        price_scenarios.append(p)
        time_scenarios.append(times)
        month_scenarios.append(months)
        forecast_scenarios.append(forecast_p)
        base_scenarios.append(base_scaled)
        flex_scenarios.append(flex_scaled)
        Delta_scenarios.append(deltas)


        if saved:
            pickle_filename = f"demand_traces/{trace_name}_month{month}_num{num_scenarios}_deadline{T}_prop{proportion_base}_CONTEXT.pkl"
            # save the generated scenarios to a pickle file for future use
            with open(pickle_filename, "wb") as f:
                pickle.dump((price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios), f)
            print(f"Saved generated scenarios to {pickle_filename}.")
        
    return price_scenarios, time_scenarios, month_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max

def load_simple_scenarios_with_flexible_context(num_scenarios, T, trace_name, proportion_base=0.5, month=1, scale_divisor=40, saved=True):
    """
    Load scenarios with base and flexible demand plus deadlines.

    Returns:
      price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_f_scenarios, p_min, p_max
    """
    price_scenarios = []
    forecast_scenarios = []
    time_scenarios = []
    month_scenarios = []
    base_scenarios = []
    flex_scenarios = []
    Delta_scenarios = []

    signal, forecasts, datetimes, p_min, p_max = load_signal_trace_with_context(trace_name, month=month)

    # if we don't already have scenarios pickled for this configuration, generate them
    # first check for a pickled file in demand_traces
    if saved:
        pickle_filename = f"demand_traces/{trace_name}_month{month}_num{num_scenarios}_deadline{T}_prop{proportion_base}_CONTEXT.pkl"
        try:
            with open(pickle_filename, "rb") as f:
                price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios = pickle.load(f)
            print(f"Loaded {num_scenarios} scenarios from {pickle_filename}.")
            return price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max
        except FileNotFoundError:
            print(f"No pickled scenarios found at {pickle_filename}, generating new scenarios.")

    
    while len(price_scenarios) < num_scenarios:
        # randomly choose an index from datetimes, and make sure there are at least T slots including/after that index
        index = np.random.randint(0, len(datetimes) - T)
        dtSequence = datetimes[index:index+T]

        # Get the signal trace for the sequence
        p = signal[dtSequence].tolist()
        # get the forecast trace for the sequence
        forecast_p = forecasts[dtSequence].tolist()

        # using the dt sequence, extract the time (hour:min) and month as context features
        times = [dt.time() for dt in dtSequence]
        months = [dt.month for dt in dtSequence]

        # randomly choose an available index from the ctx to get demand traces
        bucket_min, bucket_max = int(ctx["bucket_stats"]["bucket"].min()), int(ctx["bucket_stats"]["bucket"].max())
        # choose a random start bucket index so that we have at least T buckets available
        start_bucket = np.random.randint(bucket_min, bucket_max - T)
        idxs = list(range(start_bucket, start_bucket + T))

        ##################################### simple hardcoded demand sequence
        # if T = 48, this is 0.5 base demand arriving at times 23 and 47
        b = [0.0 for _ in range(T)]
        b[23] = 0.5
        b[47] = 0.5
        base_scaled = b

        # flexible demand of 0.4 arriving at times 0 and 23, with deadlines 23 and 47
        f = [0.0 for _ in range(T)]
        f[0] = 0.4
        f[23] = 0.4
        flex_scaled = f
        deltas = [0 for _ in range(T)]
        deltas[0] = 23
        deltas[23] = 47

        # make sure that solving for the optimal solution is feasible
        s_0 = 0.0
        x_0 = 0.0
        z_0 = 0.0
        gamma = 10 # switching cost parameter for x
        delta = 5 # switching cost parameter for z
        c_delivery = 0.2 # parameter for the discharge cost function
        eps_delivery = 0.05 # parameter for the discharge cost function (eps_delivery < c_delivery)
        S = 1 # maximum capacity of the inventory (everything scaled to unit sizes)
        try:
            status, results = opt_sol.optimal_solution(
                T, p, gamma, delta, c_delivery, eps_delivery, S, base_scaled, flex_scaled, deltas, s_0, x_0, z_0, solver_timeout=60.0
            )
            if status != "Optimal":
                print("Generated scenario is not feasible for OPT, skipping this scenario.")
                continue
        except:
            print("Error occurred while solving for OPT, skipping this scenario.")
            continue

        price_scenarios.append(p)
        time_scenarios.append(times)
        month_scenarios.append(months)
        forecast_scenarios.append(forecast_p)
        base_scenarios.append(base_scaled)
        flex_scenarios.append(flex_scaled)
        Delta_scenarios.append(deltas)


        if saved:
            pickle_filename = f"demand_traces/{trace_name}_month{month}_num{num_scenarios}_deadline{T}_prop{proportion_base}_CONTEXT.pkl"
            # save the generated scenarios to a pickle file for future use
            with open(pickle_filename, "wb") as f:
                pickle.dump((price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios), f)
            print(f"Saved generated scenarios to {pickle_filename}.")
        
    return price_scenarios, time_scenarios, month_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max


def load_isolated_scenarios_with_flexible_context(num_scenarios, trace_name, proportion_base=0.5, month=1, scale_divisor=40, saved=True):
    """
    Load scenarios with base and flexible demand plus deadlines.

    Returns:
      price_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_f_scenarios, p_min, p_max
    """
    price_scenarios = []
    forecast_scenarios = []
    time_scenarios = []
    month_scenarios = []
    base_scenarios = []
    flex_scenarios = []
    Delta_scenarios = []

    signal, forecasts, datetimes, p_min, p_max = load_signal_trace_with_context(trace_name, month=month)

    # if we don't already have scenarios pickled for this configuration, generate them
    # first check for a pickled file in demand_traces
    
    while len(price_scenarios) < num_scenarios:
        # choose T randomly between 1 and 48
        T = np.random.randint(1, 49)

        # randomly choose an index from datetimes, and make sure there are at least T slots including/after that index
        index = np.random.randint(0, len(datetimes) - T)
        dtSequence = datetimes[index:index+T]

        # Get the signal trace for the sequence
        p = signal[dtSequence].tolist()
        # get the forecast trace for the sequence
        forecast_p = forecasts[dtSequence].tolist()

        # using the dt sequence, extract the time (hour:min) and month as context features
        times = [dt.time() for dt in dtSequence]
        months = [dt.month for dt in dtSequence]

        ##################################### simple hardcoded demand sequence
        b = [0.0 for _ in range(T)]
        f = [0.0 for _ in range(T)]

        # with probability proportion_base...
        if np.random.rand() < proportion_base:
            # base demand of 1 arriving at time T
            b[-1] = 1.0
        else:
            # flexible demand of 1 arriving at time 0, with deadline T
            f[0] = 1.0

        base_scaled = b
        flex_scaled = f
        deltas = [0 for _ in range(T)]
        deltas[0] = T-1

        # make sure that solving for the optimal solution is feasible
        s_0 = 0.0
        x_0 = 0.0
        z_0 = 0.0
        gamma = 10 # switching cost parameter for x
        delta = 5 # switching cost parameter for z
        c_delivery = 0.2 # parameter for the discharge cost function
        eps_delivery = 0.05 # parameter for the discharge cost function (eps_delivery < c_delivery)
        S = 1 # maximum capacity of the inventory (everything scaled to unit sizes)
        try:
            status, results = opt_sol.optimal_solution(
                T, p, gamma, delta, c_delivery, eps_delivery, S, base_scaled, flex_scaled, deltas, s_0, x_0, z_0, solver_timeout=60.0
            )
            if status != "Optimal":
                print("Generated scenario is not feasible for OPT, skipping this scenario.")
                continue
        except:
            print("Error occurred while solving for OPT, skipping this scenario.")
            continue

        price_scenarios.append(p)
        time_scenarios.append(times)
        month_scenarios.append(months)
        forecast_scenarios.append(forecast_p)
        base_scenarios.append(base_scaled)
        flex_scenarios.append(flex_scaled)
        Delta_scenarios.append(deltas)
        
    return price_scenarios, time_scenarios, month_scenarios, forecast_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max