import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


def plot_param_evolution(param, logy=True):

    # Figure out bins for histograms
    if logy:
        bins_general = np.linspace(np.log(np.min(param)), np.log(np.max(param)), 100)
    else:
        bins_general = np.linspace(np.min(param), np.max(param), 100)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = np.zeros(0)
    y = np.zeros(0)
    c = np.zeros(0)

    iterations = param.shape[0]
    population = param.shape[1]

    # To avoid too cluttered plots
    if iterations > 200:
        iterations_range = np.linspace(0, iterations - 1, 200, dtype=int)
    else:
        iterations_range = range(iterations)

    for it in iterations_range:

        # Get strategies from agents
        if logy:
            bins, hist = np.histogram(np.log(param[it, :]), bins=bins_general)
        else:
            bins, hist = np.histogram(param[it, :], bins=bins_general)
        if logy:
            y = np.append(y, np.exp(hist[np.where(bins != 0)]))
        else:
            y = np.append(y, hist[np.where(bins != 0)])
        x = np.append(x, np.ones_like(hist[np.where(bins != 0)]) * it)
        c = np.append(c, bins[np.where(bins != 0)] / np.sum(bins))

    ax.grid()

    histograms = ax.scatter(x, y, c=c, cmap='hot')

    # Plot population average
    x_avg = np.linspace(0, iterations, iterations)
    y_avg = np.mean(param, axis=1)
    if logy:
        plt.semilogy(x_avg, y_avg, color='tab:green', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
                     lw=2, label='Population average', alpha=0.7)
    else:
        plt.plot(x_avg, y_avg, color='tab:green', path_effects=[pe.Stroke(linewidth=5, foreground='w'), pe.Normal()],
                 lw=2, label='Population average', alpha=0.7)

    ax.set_xlabel('Evolution iteration')
    cbar = fig.colorbar(histograms)
    cbar.ax.set_ylabel('Population percentage (N={})'.format(population), rotation=270, labelpad=15)

    return ax
