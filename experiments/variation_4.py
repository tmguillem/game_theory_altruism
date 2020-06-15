from genetic_algorithm import GA
from experiments.plots import plot_param_evolution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def main():
    """
    This function runs variation 4.
    In variation 4, the population of individuals is generated with randomly assigned strategies x (different for every
    agent). Furthermore, all players are randomly assigned an alpha parameter at random within [0.5, 1], and we let the
    strategies x and the parameters alpha evolve
    """

    population = 200
    iterations = 2000
    k = 0.5
    m = 10
    mu = 0.1
    mutable_params = ['alpha']
    x = None
    alpha = None

    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=x, alpha_init=alpha,
                      mutable_parameters=mutable_params)

    summary = genetic_algo.run()
    print(np.mean(summary['x'][-1, :]))
    ax = plot_param_evolution(summary['x'])
    ax.set_ylabel('log(x) histogram')
    ax.legend()
    ax.set_title('Variation 4 experiment - x evolution')

    ax = plot_param_evolution(summary['alpha'], logy=False)
    ax.set_ylabel('alpha histogram')
    ax.legend()
    ax.set_title('Variation 4 experiment - alpha evolution')

    plt.show()


if __name__ == '__main__':
    main()