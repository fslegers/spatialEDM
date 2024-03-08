import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import pearsonr
import numpy as np

def make_plot_per_obs_noise(df, test="initial point"):

    # Create a list of colors for the different noise levels
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(0, len(set(df['obs_noise']))-1)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [scalar_map.to_rgba(value) for value in range(len(set(df['obs_noise'])))]

    for horizon in set(df['hor']):
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        c = 0

        for obs_noise_level in set(df['obs_noise']):
            df_sub = df[(df['obs_noise'] == obs_noise_level) & (df['hor'] == horizon)]
            levels, simplex_corr, smap_corr = [], [], []
            variance_levels = sorted(list(set(df_sub['var'])))

            for variance in variance_levels:
                x = df_sub[df_sub['var'] == variance]['obs'].values
                y1 = df_sub[df_sub['var'] == variance]['pred_simplex'].values
                y2 = df_sub[df_sub['var'] == variance]['pred_smap'].values

                # Remove NA values
                x1, simplex, x2, smap, = [], [], [], []
                for i in range(len(x)):
                    if not(np.isnan(x[i]) | np.isnan(y1[i])):
                        x1.append(x[i])
                        simplex.append(y1[i])
                    if not(np.isnan(x[i]) | np.isnan(y2[i])):
                        x2.append(x[i])
                        smap.append(y2[i])

                # Calculate Pearson correlation coefficient
                try:
                    r1, p1 = pearsonr(x1, simplex)
                    r2, p2 = pearsonr(x1, smap)
                    levels.append(variance)
                    simplex_corr.append(r1)
                    smap_corr.append(r2)
                except:
                    time.sleep(1)

            axes[0].plot(levels, simplex_corr, color=colors[c])
            axes[1].plot(levels, smap_corr, color=colors[c])
            axes[0].scatter(levels, simplex_corr, color=colors[c])
            axes[1].scatter(levels, smap_corr, color=colors[c])
            c += 1

        fig.suptitle("Horizon = " + str(horizon))
        plt.show()

    return 0

def make_plot_per_horizon(df, test="initial point"):
    return 0

if __name__ == "__main__":

    # Change directory
    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    # Load initial point variance data
    df = pd.read_csv("initial point variance, n_iterations = 25, n_replicates = 25, ts_length = 50, hor = 10.csv")
    df = df.rename(columns={"init_var": "var"})

    make_plot_per_obs_noise(df, "initial point")
    #make_plot_per_horizon(df, "initial point")

    # Load rho variance data
    df = pd.read_csv("initial point variance, n_iterations = 25, n_replicates = 25, ts_length = 50, hor = 10.csv")
    df = df.rename(columns={"rho_var": "var"})

    make_plot_per_obs_noise(df, "initial point")
    #make_plot_per_horizon(df, "initial point")

    # Change directory
    os.chdir('..')
    os.chdir('figures')