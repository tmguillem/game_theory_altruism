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

    # Change to 3 or 4 for variations 3 or 4
    variation = 3

    assert variation == 3 or variation == 4

    population = 100
    iterations = 1000
    k = 0.5
    m = 0.5
    mu = 0.01
    alpha = None if variation == 4 else 0.5
    mutable_params = ['alpha','x'] if variation == 4 else ['x']

    x = None

    genetic_algo = GA(n_population=population, m_iterations=iterations,
                      k=k, m=m, mu=mu,
                      x_init=x, alpha_init=alpha,
                      mutable_parameters=mutable_params)

    summary = genetic_algo.run()

    # Figure out bins for histograms
    bins_general = np.linspace(np.log(np.min(summary['alpha'])), np.log(np.max(summary['alpha'])), 100) if variation == 4 else np.linspace(np.log(np.min(summary['x'])), np.log(np.max(summary['x'])), 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)
   
    x = np.zeros(0)
    y = np.zeros(0)
    c = np.zeros(0)

    for it in range(iterations + 1):

        # Get strategies from agents
        bins, hist = np.histogram(np.log(summary['alpha'][it, :]), bins=bins_general) if variation == 4 else np.histogram(np.log(summary['x'][it, :]), bins=bins_general)
        y = np.append(y, np.exp(hist[np.where(bins != 0)]))
        x = np.append(x, np.ones_like(hist[np.where(bins != 0)]) * it)
        c = np.append(c, bins[np.where(bins != 0)] / np.sum(bins))

    ax.grid()

    histograms = ax.scatter(x, y, c=c, cmap='hot', edgecolors='k')

    # Plot theoretical value
    xlim = ax.get_xlim()
    if variation == 3:
        
        x_tilde = np.linspace(xlim[0], xlim[1], 100)
        y_tilde = (alpha*m)/(2*alpha-k)* np.ones_like(x_tilde)
        plt.semilogy(x_tilde, y_tilde, color='tab:blue', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
                 lw=2, label='Theoretical value', alpha=0.7)

    # Plot population average
    x_avg = np.linspace(0, iterations + 1, iterations + 1)
    y_avg = np.mean(summary['alpha'], axis=1) if variation == 4 else np.mean(summary['x'], axis=1)
    plt.semilogy(x_avg, y_avg, color='tab:green', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
                 lw=2, label='Population average', alpha=0.7)
    ax.legend()
    ax.set_xlim(xlim)
   
    ax.set_ylabel('log(alpha) histogram' if variation == 4 else 'log(x) histogram')
    ax.set_xlabel('Evolution iteration')
    ax.set_title('Variation 4 experiment' if variation == 4 else 'Variation 3 experiment')
    cbar = fig.colorbar(histograms)
    cbar.ax.set_ylabel('Population percentage (N={})'.format(population), rotation=270, labelpad=15)

    plt.show()


if __name__ == '__main__':
    main()


