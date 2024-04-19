import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from src.classes import *
from src.simulate_lorenz import *


def plot_RMSE_vs_horizon(df, rho):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
        df_sub = df[df['training_length'] == len]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 2.0, 4.0]:
            color = colors(i)
            marker = markers[i]
            df_noise = df_sub[df_sub['noise'] == noise]
            plt.plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color)
            plt.scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker)
            plt.fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
            #plt.plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linestyle="--")
            #plt.plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('horizon')
        plt.ylabel('RMSE')
        #plt.title(f'len = {len}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/RMSE, horizon/len={len}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

def plot_corr_vs_horizon(df, rho):
    colors = plt.get_cmap('tab10')

    for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
        df_sub = df[df['training_length'] == len]
        plt.figure()

        # Iterate over groups and plot corr
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            color = colors(i)
            df_noise = df_sub[df_sub['noise'] == noise]
            plt.plot(df_noise['hor'], df_noise['mean_corr'], label=f'{noise}', color=color)
            plt.scatter(df_noise['hor'], df_noise['mean_corr'], color=color)
            plt.plot(df_noise['hor'], df_noise['min_corr'], color=color, linestyle="--")
            plt.plot(df_noise['hor'], df_noise['max_corr'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('horizon')
        plt.ylabel('corr')
        plt.title(f'len = {len}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/corr, horizon/len={len}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

def plot_RMSE_vs_noise(df, rho):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v', 'p']

    #for hor in range(1, 11):
    for hor in [1]:
        df_sub = df[df['hor'] == hor]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for len in [25, 50, 75, 100]:
            color = colors(i)
            marker = markers[i]
            df_len = df_sub[df_sub['training_length'] == len]
            plt.plot(df_len['noise'], df_len['mean_RMSE'], label=f'{len}', color=color)
            plt.scatter(df_len['noise'], df_len['mean_RMSE'], color=color)
            plt.plot(df_len['noise'], df_len['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_len['noise'], df_len['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Noise')
        plt.ylabel('RMSE')
        #plt.title(f'hor = {hor}')
        plt.legend(title='Training length')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/RMSE, noise/hor={hor}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

def plot_corr_vs_noise(df, rho):
    colors = plt.get_cmap('tab10')

    for hor in range(1, 11):
        df_sub = df[df['hor'] == hor]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
            color = colors(i)
            df_len = df_sub[df_sub['training_length'] == len]
            plt.plot(df_len['noise'], df_len['mean_corr'], label=f'{len}', color=color)
            plt.scatter(df_len['noise'], df_len['mean_corr'], color=color)
            plt.plot(df_len['noise'], df_len['min_corr'], color=color, linestyle="--")
            plt.plot(df_len['noise'], df_len['max_corr'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Noise')
        plt.ylabel('corr')
        plt.title(f'hor = {hor}')
        plt.legend(title='Training length')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/corr, noise/hor={hor}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

def plot_RMSE_vs_length(df, rho):
    colors = plt.get_cmap('tab10')

    for hor in range(1, 11):
        df_sub = df[df['hor'] == hor]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            color = colors(i)
            df_len = df_sub[df_sub['noise'] == noise]
            plt.plot(df_len['training_length'], df_len['mean_RMSE'], label=f'{len}', color=color)
            plt.scatter(df_len['training_length'], df_len['mean_RMSE'], color=color)
            plt.plot(df_len['training_length'], df_len['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_len['training_length'], df_len['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Training length')
        plt.ylabel('RMSE')
        plt.title(f'hor = {hor}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/RMSE, length/hor={hor}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

def plot_corr_vs_length(df, rho):
    colors = plt.get_cmap('tab10')

    for hor in range(1, 11):
        df_sub = df[df['hor'] == hor]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            color = colors(i)
            df_len = df_sub[df_sub['noise'] == noise]
            plt.plot(df_len['training_length'], df_len['mean_corr'], label=f'{noise}', color=color)
            plt.scatter(df_len['training_length'], df_len['mean_corr'], color=color)
            plt.plot(df_len['training_length'], df_len['min_corr'], color=color, linestyle="--")
            plt.plot(df_len['training_length'], df_len['max_corr'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Training length')
        plt.ylabel('corr')
        plt.title(f'hor = {hor}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/corr, length/hor={hor}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

def plot_6(df_28, df_20):

    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']
    plt.rc('font', size=16)

    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    plt.subplots_adjust(hspace=1.5)

    df = df_28
    # First plot
    # Length = 25, RMSE vs horizon
    df_sub = df[df['training_length'] == 25]

    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df_sub[df_sub['noise'] == noise]
        axs[0,0].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
        axs[0,0].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[0,0].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[0, 0].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
        axs[0, 0].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
        i += 1

    #axs[0,0].set_xlabel('horizon)
    axs[0,0].set_ylabel('RMSE')

    # Second plot
    # Length = 75, RMSE vs horizon
    df_sub = df[df['training_length'] == 75]

    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df_sub[df_sub['noise'] == noise]
        axs[0, 1].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
        axs[0, 1].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[0, 1].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[0, 1].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
        axs[0, 1].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
        i += 1

    #axs[0, 1].set_xlabel('horizon)
    #axs[0, 1].set_ylabel('RMSE')

    # third plot
    # RMSE vs length
    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df[(df['noise'] == noise) & (df['hor'] == 1)]
        axs[0, 2].fill_between(df_noise['training_length'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
        axs[0, 2].plot(df_noise['training_length'], df_noise['min_RMSE'], linewidth=1, alpha=.8)
        axs[0, 2].plot(df_noise['training_length'], df_noise['max_RMSE'], linewidth=1, alpha=.8)
        axs[0, 2].plot(df_noise['training_length'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
        axs[0, 2].scatter(df_noise['training_length'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
        i += 1

    # Set labels and title
    #axs[0, 2].set_xlabel('Training length')
    #axs[0, 2].set_ylabel('RMSE')

    df = df_20
    # 4th plot
    # Length = 25, RMSE vs horizon
    df_sub = df[df['training_length'] == 25]

    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df_sub[df_sub['noise'] == noise]
        axs[1, 0].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
        axs[1, 0].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[1, 0].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[1, 0].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
        axs[1, 0].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
        i += 1

    axs[1, 0].set_xlabel('horizon')
    axs[1, 0].set_ylabel('RMSE')

    # 5th plot
    # Length = 75, RMSE vs horizon
    df_sub = df[df['training_length'] == 75]

    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df_sub[df_sub['noise'] == noise]
        axs[1, 1].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
        axs[1, 1].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[1, 1].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[1, 1].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
        axs[1, 1].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
        i += 1

    axs[1, 1].set_xlabel('horizon')
    #axs[1, 1].set_ylabel('RMSE')

    # 6th plot
    # RMSE vs length
    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df[(df['noise'] == noise) & (df['hor'] == 1)]
        axs[1, 2].fill_between(df_noise['training_length'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1,color=color)
        axs[1, 2].plot(df_noise['training_length'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[1, 2].plot(df_noise['training_length'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
        axs[1, 2].plot(df_noise['training_length'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
        axs[1, 2].scatter(df_noise['training_length'], df_noise['mean_RMSE'], color=color, marker=marker, s=50)
        i += 1

    # Set labels and title
    axs[1, 2].set_xlabel('training length')
    #axs[1, 2].set_ylabel('RMSE')


    # Add (a) - (f)
    axs[0, 0].text(0.5, -0.25, '(a)', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.5, -0.25, '(b)', transform=axs[0, 1].transAxes)
    axs[0, 2].text(0.5, -0.25, '(c)', transform=axs[0, 2].transAxes)
    axs[1, 0].text(0.5, -0.35, '(d)', transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.5, -0.35, '(e)', transform=axs[1, 1].transAxes)
    axs[1, 2].text(0.5, -0.35, '(f)', transform=axs[1, 2].transAxes)

    axs[0, 0].text(-0.3, 0.37, r'$\rho = 28$', transform=axs[0, 0].transAxes, fontsize=16, rotation=90)
    axs[1, 0].text(-0.3, 0.37, r'$\rho = 20$', transform=axs[1, 0].transAxes, fontsize=16, rotation=90)

    fig.tight_layout(pad=3)

    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/6 plots.png"
    plt.savefig(path, dpi = 900)
    plt.show()


if __name__ == '__main__':

    # Load CSV file into a pandas DataFrame
    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = 28/mean and percentiles.csv"
    df_28 = pd.read_csv(path)

    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = 20/mean and percentiles.csv"
    df_20 = pd.read_csv(path)

    plot_6(df_28, df_20)







