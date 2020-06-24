from genetic_algorithm import GA
from experiments.plots import plot_param_evolution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib


def main():
    """
    This function runs variation 3.
    In variation 3, the population of individuals is generated with randomly assigned strategies x (different for every
    agent). Furthermore, all players are equally non-egoistic, i.e. 1/2 <= alpha < 1.
    We let the strategies x evolve
    """
    font = {'family': 'normal', 'size': 15}
    matplotlib.rc('font', **font)

    population = 200
    iterations = 250
    k = 0.5
    m = 0.5
    mu = 0.1
    alpha = 0.8
    mutable_params = ['x']
    x = None

    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=x, alpha_init=alpha,
                      mutable_parameters=mutable_params)

    alpha = 0.75 if alpha is None else alpha

    summary, _ = genetic_algo.run()
    print(np.mean(summary['x'][-1, :]))
    ax = plot_param_evolution(summary['x'])

    # Plot theoretical value (equation 12) -> Maximizes player success.
    xlim = ax.get_xlim()
    x_tilde = np.linspace(xlim[0], xlim[1], 100)
    y_tilde = (alpha * m * (2 * alpha + k)) / (4 * alpha * alpha - k ** 2) * np.ones_like(x_tilde)
    plt.semilogy(x_tilde, y_tilde, color='tab:blue', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
                 lw=2, label='Theoretical value', alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylabel('log(x) histogram')
    ax.legend()
    ax.set_title('Altruistic equilibrium')

    ax = plot_param_evolution(summary['alpha'], logy=False)

    plt.show()


if __name__ == '__main__':
    main()
