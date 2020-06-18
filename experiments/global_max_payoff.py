from genetic_algorithm import GA
from experiments.plots import plot_param_evolution, iterate_ga, calc_optimal_payoff, plot_payoff_convergence, plot_prop1_convergence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def main():
    
    # 1. Verify convergence to global maximum over different parameter values
    # Mutation in altruism allows population to move to a utility maximising frontier
    vals, summary_list = iterate_ga()
    
    iters = len(summary_list)
    act_u = np.zeros(iters)
    exp_u = np.zeros(iters)
    
    for i in range(iters):
        mean_u = np.mean(summary_list[i]['u'],axis=1)
        act_u[i] = np.mean(mean_u[-100:])
        exp_u[i] = calc_optimal_payoff(m=vals[i,1], k=vals[i,0])
    
    plot_payoff_convergence(act_u, exp_u)
    
    # 2. Verify proposition 1 for a fixed m
    # Population of altruistic players reaches higher success than egoists
    # Differences are greater the larger the absolute value of k
    vals_alt, summary_list_alt = iterate_ga(vals_m = np.array(0.7), alpha_init = None,
                                            mutable_parameters=['alpha'], num = 50)
    vals_ego, summary_list_ego = iterate_ga(vals_m = np.array(0.7), alpha_init = 1,
                                            mutable_parameters=[], num = 50)
    
    iters = len(vals_alt)
    vals_k = vals_alt[:,0]
    diff_u = np.zeros(iters)
    
    for i in range(iters):
        u_alt = np.mean(np.mean(summary_list_alt[i]['u'], axis=1)[-100:])
        u_ego = summary_list_ego[i]['u'][-1,0]
        diff_u[i] = u_alt - u_ego
        
    
    plot_prop1_convergence(vals_k, diff_u)
    
    # 3. Verify proposition 2
    

    
    
if __name__ == '__main__':
    main()
    