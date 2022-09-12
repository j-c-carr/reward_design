import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_of_trajectory_distribution(fname, H_T, rho, color='b'):
    """Plots the distribution of trajectories for a given policy"""

    fig, ax = plt.subplots(figsize=(6, 20))
    # Label each trajectory
    y = [i for i in range(rho.shape[0])]
    labels = [str(H_T[i]) for i in y]

    ax.barh(y, rho, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    #ax.tick_params(axis='y', labelrotation=90)
    ax.invert_yaxis()
    plt.savefig(fname, dpi=400, bbox_inches='tight')
