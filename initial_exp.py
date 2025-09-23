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
    T = 96 # consider a 24-hour time horizon, 15-min price intervals

    gamma = 10 # switching cost parameter for x
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
    print(f"Loaded {num_scenarios} scenarios with T={T} from trace {trace_name}.")

    # for now just use the first scenario
    p = price_scenarios[0]
    b = base_scenarios[0]
    f = flex_scenarios[0]
    Delta_f = Delta_scenarios[0]


    ############################################# solve for the different technique solutions
    # --- Optimal Solution ---
    status, results = opt_sol.optimal_solution(
        T, p, gamma, delta, c_delivery, eps_delivery, S, b, f, Delta_f
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
        opt_obj = pi.objective_function(T, p, gamma, delta, c_delivery, eps_delivery, results['x'], results['z'])
        print("Optimal objective value:", opt_obj)
        print("Total charge (sum x):", sum(results['x']))
        print("Total discharge (sum z):", sum(results['z']))
    else:
        print("No optimal solution found. Status:", status)

    # --- Online PAAD Solution ---
    results = pi.paad_algorithm(
        T, p, gamma, delta, c_delivery, eps_delivery, p_min, p_max, S, b, f, Delta_f
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
