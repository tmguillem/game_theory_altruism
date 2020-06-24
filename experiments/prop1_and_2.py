from genetic_algorithm import GA
from experiments.plots import iterate_ga, plot_payoff_convergence, plot_prop1_convergence, plot_prop2_convergence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def main():
    font = {'family': 'normal', 'size': 15}
    matplotlib.rc('font', **font)

    # 1. Verify convergence to global maximum over different parameter values
    # Mutation in altruism allows population to move to a utility maximising frontier
    vals, summary_list = iterate_ga(mutable_parameters=['alpha'])

    iters = len(summary_list)
    act_u = np.zeros(iters)
    exp_u = np.zeros(iters)
    act_x = np.zeros(iters)
    exp_x = np.zeros(iters)
    
    for i in range(iters):
        mean_u = np.mean(summary_list[i]['u'], axis=1)
        mean_x = np.mean(summary_list[i]['x'], axis=1)
        act_u[i] = np.mean(mean_u[-100:])
        act_x[i] = np.mean(mean_x[-100:])

        # Equation 4: theoretical strategy x that maximizes joint payoff
        exp_u[i] = (vals[i, 1] ** 2) / (4 * (1 - vals[i, 0]))
        exp_x[i] = vals[i, 1] / (2 - 2 * vals[i, 0])

    plot_payoff_convergence(act_u, exp_u, act_x, exp_x)
    
    # 2. Verify proposition 1 for a fixed m
    # Population of altruistic players reaches higher success than egoists
    # Differences are greater the larger the absolute value of k

    # Keep m value constant
    m = 10
    vals_alt, summary_list_alt = iterate_ga(vals_m=np.array(m), alpha_init=None, mutable_parameters=['alpha'], num=50)
    vals_ego, summary_list_ego = iterate_ga(vals_m=np.array(m), alpha_init=1, mutable_parameters=[], num=50)

    iters = len(vals_alt)
    vals_k = vals_alt[:, 0]
    diff_u = np.zeros(iters)

    for i in range(iters):
        u_alt = np.mean(np.mean(summary_list_alt[i]['u'], axis=1)[-100:])
        u_ego = summary_list_ego[i]['u'][-1, 0]
        diff_u[i] = u_alt - u_ego

    plot_prop1_convergence(vals_k, diff_u, m)
    
    # 3. Verify proposition 2
    # The payoff of an egoist player in an altruist-egoistic interaction is always greater
    # The payoff difference also increases with the difference in the altruism parameter
    population = 200
    iterations = 200
    k = 0.3
    m = 0.7
    mu = 0.1
    mutable_params = ['alpha']
    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=None, alpha_init=None,
                      mutable_parameters=mutable_params)
    summary, pairings_summary = genetic_algo.run()
    
    plot_prop2_convergence(population=population,
                           pairings_summary=pairings_summary,
                           summary=summary,
                           iterations=iterations,
                           used_k=k, used_m=m)
    
    plt.show()


if __name__ == '__main__':
    main()
