# import other code
import paad_implementation as pi
import opt_sol as opt_sol
# import base libraries
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import seaborn as sns
import pickle
import sys
from tqdm import tqdm
from load_signal_trace import load_signal_trace
from functions import load_scenarios_with_flexible

np.set_printoptions(suppress=True,precision=3)

def experiment():
    ############################################# set up OSDM experiment parameters
    T = 48 # consider a 24-hour time horizon, 15-min price intervals

    eta = 10 # tracking cost parameter for x
    delta = 5 # switching cost parameter for z

    c_delivery = 0.2 # parameter for the discharge cost function
    eps_delivery = 0.05 # parameter for the discharge cost function (eps_delivery < c_delivery)

    S = 1 # maximum capacity of the inventory (everything scaled to unit sizes)

    # use the function to load scenarios
    num_scenarios = 1
    trace_name = "CAISO"

    price_scenarios, base_scenarios, flex_scenarios, Delta_scenarios, p_min, p_max = load_scenarios_with_flexible(
        num_scenarios, T, trace_name, proportion_base=0.5
    )

    # set tracking targets to be the total demand (base + flex) evenly spread over T slots
    base_all = base_scenarios
    flex_all = flex_scenarios
    price_all = price_scenarios
    tracking_target_all = [[0.0 for _ in range(T)] for _ in range(num_scenarios)]
    for i in range(num_scenarios):
        total_demand = sum(base_all[i]) + sum(flex_all[i])
        even_target = total_demand / T
        tracking_target_all[i] = [even_target for _ in range(T)]
        # choose random time slots to have zero target -- probability weighted by the price (higher price, higher chance of zero target)
        price_seq = np.array(price_all[i])
        probs = price_seq / np.sum(price_seq)
        rng = np.random.default_rng(42)
        # choose a random number of indexes to pick: 2, 3, or 4
        num_indexes = rng.choice([2, 3, 4])
        chosen_indexes = rng.choice(int(T), size=num_indexes, replace=False, p=probs)
        reallocation = 0.0
        for idx in chosen_indexes:
            reallocation += tracking_target_all[i][idx]
            tracking_target_all[i][idx] = 0.0
        # reallocate the removed target evenly to other slots
        reallocation_per_slot = reallocation / (T - num_indexes)
        for t in range(T):
            if t not in chosen_indexes:
                tracking_target_all[i][t] += reallocation_per_slot
    
    print(f"Loaded {num_scenarios} scenarios with T={T} from trace {trace_name}.")

    # print the p_min and p_max values
    print(f"p_min: {p_min}, p_max: {p_max}")

    # for now just use the first scenario
    p = price_scenarios[0]
    b = base_scenarios[0]
    f = flex_scenarios[0]
    Delta_f = Delta_scenarios[0]
    a = tracking_target_all[0]


    ############################################# solve for the different technique solutions
    # --- Optimal Solution ---
    status, results = opt_sol.optimal_tracking_solution(
        T, p, eta, delta, c_delivery, eps_delivery, S, b, f, Delta_f, a
    )
    
    # if status == "Optimal":
    #     print("Optimal solution found.")
    #     print("Objective value:", results['obj_val'])
    #     print("x:     ", results['x'])
    #     print("z:     ", results['z'])
    #     print("s:", results['s'])
    # else:
    #     print("No optimal solution found. Status:", status)
    # print optimal objective value, sum(x), and sum(z)
    if status == "Optimal":
        # recompute the objective value for this solution
        opt_obj = pi.tracking_objective_function(T, p, eta, delta, c_delivery, eps_delivery, results['x'], results['z'], a)
        print("Optimal objective value:", opt_obj)
        print("Total charge (sum x):", sum(results['x']))
        print("Total discharge (sum z):", sum(results['z']))
    else:
        print("No optimal solution found. Status:", status)

    # --- Online PAAD Solution ---
    results = pi.paad_algorithm(
        T, p, eta, delta, c_delivery, eps_delivery, p_min, p_max, S, b, f, Delta_f
    )
    # if results is not None:
    #     print("PAAD solution found.")
    #     print("Objective value:", results['obj_val'])
    #     print("x:     ", results['x'])
    #     print("z:     ", results['z'])
    #     print("s:", results['s'])
    # else:
    #     print("No PAAD solution found.")
    # print optimal objective value, sum(x), and sum(z)
    if results is not None:
        print("PAAD objective value:", results['obj_val'])
        print("Total charge (sum x):", sum(results['x']))
        print("Total discharge (sum z):", sum(results['z']))
        # print the discharge decisions at each time step
        print("Discharge decisions (z) at each time step:", results['z'])
    else:
        print("No PAAD solution found.")


    return None






# we will be able to use multiprocessing here eventually
if __name__ == "__main__":
    experiment()
