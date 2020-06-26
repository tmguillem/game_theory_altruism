from genetic_algorithm import GA
from plots import plot_param_evolution
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

    population = 500
    iterations = 5000
    k = 0.5
    m = 0.5
    mu = 0.01
    mutable_params = ['x','alpha']
    x = None
    alpha = None
    reprod_method = "payoff"
    rational = True
    reciprocal = True


    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=x, alpha_init=alpha,
                      mutable_parameters=mutable_params,reprod_method = reprod_method,rational = rational,reciprocal = reciprocal)

    summary = genetic_algo.run()
    print(np.mean(summary['x'][-1, :]))

    ax = plot_param_evolution(summary['x'])
    xlim = ax.get_xlim()
    x_tilde = np.linspace(xlim[0], xlim[1], 100)
    y_tilde = m/(2-k) * np.ones_like(x_tilde)
    plt.plot(x_tilde, y_tilde, color='tab:blue', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
             lw=2, label='Theoretical value', alpha=0.7)
    ax.set_ylabel('log(x) histogram')
    ax.legend()
    ax.set_title('x evolution')

    ax = plot_param_evolution(summary['alpha'], logy=False)
    xlim = ax.get_xlim()
    #x_tilde = np.linspace(xlim[0], xlim[1], 100)
    #y_tilde = 1 * np.ones_like(x_tilde)
    #plt.plot(x_tilde, y_tilde, color='tab:blue', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
           #  lw=2, label='Theoretical value', alpha=0.7)
    ax.set_ylabel('alpha histogram')
    ax.legend()
    ax.set_title('Alpha evolution')

    plt.show()


if __name__ == '__main__':
    main()

