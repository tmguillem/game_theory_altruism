from genetic_algorithm import GA
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


def main():
    """
    This function runs variation 1 or 2.
    In variation 1, the population of individuals is generated with the same initial strategy x, which is chosen.
    In variation 2, the population starts with randomly assigned strategies x (different for every agent).
    In both cases, we contemplate egoistic players, i.e.: Alpha=1, and only let the strategies x evolve.
    """

    # Change to 1 or 2 for variations 1 or 2
    variation = 1

    assert variation == 1 or variation == 2

    population = 50
    iterations = 100
    k = 0.5
    m = 0.5
    mu = 0.1
    alpha = 1
    mutable_params = ['x']

    x = 0.9 if variation == 1 else None

    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=x, alpha_init=alpha,
                      mutable_parameters=mutable_params)

    summary = genetic_algo.run()

    # Figure out bins for histograms
    bins_general = np.linspace(np.min(summary['x']), np.max(summary['x']), 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.zeros(0)
    y = np.zeros(0)
    c = np.zeros(0)

    for it in range(iterations + 1):

        # Get strategies from agents
        bins, hist = np.histogram(summary['x'][it, :], bins=bins_general)
        y = np.append(y, hist[np.where(bins != 0)])
        x = np.append(x, np.ones_like(hist[np.where(bins != 0)]) * it)
        c = np.append(c, bins[np.where(bins != 0)] / np.sum(bins))

    ax.grid()

    histograms = ax.scatter(x, y, c=c, cmap='hot', edgecolors='k')

    # Plot theoretical value
    xlim = ax.get_xlim()
    x_tilde = np.linspace(xlim[0], xlim[1], 100)
    y_tilde = m / (2 - k) * np.ones_like(x_tilde)
    plt.plot(x_tilde, y_tilde, color='tab:blue', lw=2, path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
             label='Theoretical value', alpha=0.7)
    ax.legend()
    ax.set_xlim(xlim)

    ax.set_ylabel('Strategy histogram')
    ax.set_xlabel('Evolution iteration')
    ax.set_title('Variation 1 experiment' if variation == 1 else 'Variation 2 experiment')
    cbar = fig.colorbar(histograms)
    cbar.ax.set_ylabel('Population percentage (N={})'.format(population), rotation=270, labelpad=15)

    plt.show()


if __name__ == '__main__':
    main()
