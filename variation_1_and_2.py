from genetic_algorithm import GA
from plots import plot_param_evolution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def main():
    """
    This function runs variation 1 or 2.
    In variation 1, the population of individuals is generated with the same initial strategy x, which is chosen.
    In variation 2, the population starts with randomly assigned strategies x (different for every agent).
    In both cases, we contemplate egoistic players, i.e.: Alpha=1, and only let the strategies x evolve.
    """

    # Change to 1 or 2 for variations 1 or 2
    variation = 2

    assert variation == 1 or variation == 2

    population = 200
    iterations = 250
    k = 0.5
    m = 0.5
    mu = 0.1
    alpha = 1
    mutable_params = ['x']
    reprod_method="utility"
    rational = False

    x = 0.9 if variation == 1 else None

    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=x, alpha_init=alpha,
                      mutable_parameters=mutable_params,reprod_method=reprod_method, rational = rational)

    summary = genetic_algo.run()
    print(np.mean(summary['x'][-1, :]))

    ax = plot_param_evolution(summary['x'])

    # Plot theoretical value (equation 7) -> Maximizes player success.
    xlim = ax.get_xlim()
    x_tilde = np.linspace(xlim[0], xlim[1], 100)
    y_tilde = m / (2 - k) * np.ones_like(x_tilde)
    plt.semilogy(x_tilde, y_tilde, color='tab:blue', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
                 lw=2, label='Theoretical value', alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylabel('log(x) histogram')
    ax.legend()
    ax.set_title('Variation 1 experiment' if variation == 1 else 'Variation 2 experiment')

    ax = plot_param_evolution(summary['u'], logy=False)
    # Plot theoretical value (equation 7) -> Maximizes player success.
    xlim = ax.get_xlim()
    x_tilde = np.linspace(xlim[0], xlim[1], 100)
    y_tilde = m ** 2 / (2 - k) ** 2 * np.ones_like(x_tilde)
    plt.plot(x_tilde, y_tilde, color='tab:blue', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
             lw=2, label='Theoretical value', alpha=0.7)
    ax.set_xlim(xlim)
    ax.set_ylabel('U histogram')
    ax.legend()
    ax.set_title('Variation 1 experiment' if variation == 1 else 'Variation 2 experiment')

    plt.show()


if __name__ == '__main__':
    main()
