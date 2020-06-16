import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import numpy as np
from genetic_algorithm import GA



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

def iterate_ga(mutable_parameters, vals_k=None, vals_m=None, x_init=None, alpha_init=None, num = 10):
    """
    Iterate over the genetic algorithm with varying k and m parameters
    :return: a list of summaries of the GA
    """
    if vals_k is None:    
        vals_k = np.linspace(-0.99,0.99, num=num)
    if vals_m is None:    
        vals_m = np.logspace(-10,10, base = 2, num=num)
    vals = np.dstack(np.meshgrid(vals_k, vals_m)).reshape(-1, 2)
    n_vals = len(vals)
    summary_list = []
    
    for i in range(n_vals):
        genetic_algo = GA(n_population=100, m_iterations=200,
                          k=vals[i,0], m=vals[i,1], mu=0.1, 
                          x_init=x_init, alpha_init=alpha_init,
                          mutable_parameters=mutable_parameters)
        summary, _ = genetic_algo.run()
        summary_list.append(summary)
    
    return vals, summary_list


def calc_optimal_payoff(m, k):
    return (m**2)/(4*(1-k))

def plot_payoff_convergence(act, exp):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=2)
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    ax.scatter(x=act, y=exp)
    ax.set_xlabel('Mean actual utility')
    ax.set_ylabel('Expected utility')
    
    return ax
    
def plot_prop1_convergence(vals_k, diff_u):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.set_yscale('log', basey=2)
    #ax.set_yscale('symlog')
    ax.plot(vals_k, diff_u, '-o')
    ax.set_xlabel('k')
    ax.set_ylabel('U_alt - U_ego')

    
    return 

def plot_prop2_convergence(population, n_interactions, pairings_summary, summary, iterations):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlim([0, 0.5])
    ax.set_xlabel('beta - alpha')
    ax.set_ylabel('U_egoist - U_altruist')
    ymax = 0
    
    n_interactions = int(population/2)
    a = np.zeros(n_interactions)
    b = np.zeros(n_interactions)
    u1 = np.zeros(n_interactions)
    u2 = np.zeros(n_interactions)
    
    #for i in range(iterations):
    for i in range(iterations):
        p1 = pairings_summary[i,:,0].astype(int)
        p2 = pairings_summary[i,:,1].astype(int)
        a_p1 = summary['alpha'][i+1,:][p1]
        a_p2 = summary['alpha'][i+1,:][p2]
        u_p1 = summary['u'][i+1,:][p1]
        u_p2 = summary['u'][i+1,:][p2]
        
        for j in range(n_interactions):
            if a_p1[j] < a_p2[j]:
                a[j] = a_p1[j]
                b[j] = a_p2[j]
                u1[j] = u_p1[j]
                u2[j] = u_p2[j]
            else:
                a[j] = a_p2[j]
                b[j] = a_p1[j]
                u1[j] = u_p2[j]
                u2[j] = u_p1[j]
        if max(u2-u1) > ymax:
            ymax = max(u2-u1)
        ax.set_ylim([0, ymax])
        ax.scatter(b-a, (u2-u1), c='black')
    
    return ax


