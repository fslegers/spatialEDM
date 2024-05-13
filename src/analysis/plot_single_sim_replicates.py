import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from src.classes import *
from src.simulate_lorenz import *
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import os


def plot_RMSE_vs_horizon(df, rho):
    plt.rc('font', size=14)
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for len in df['training_length'].unique():
        df_sub = df[df['training_length'] == len]
        fig, ax1 = plt.subplots()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in df['noise'].unique():

            color = colors(i)
            marker = markers[i]

            df_noise = df_sub[df_sub['noise'] == noise]
            ax1.plot(df_noise['hor'], df_noise['mean_RMSE'], color=color)
            ax1.scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, label=f'{noise}', marker=marker, s=50)

            ax1.fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color, zorder=1)
            ax1.plot(df_noise['hor'], df_noise['max_RMSE'], linestyle="--", dashes=(5,5), color=color, alpha=.5)
            ax1.plot(df_noise['hor'], df_noise['min_RMSE'], linestyle="--", dashes=(5,5), color=color, alpha=.5)

            i += 1

        # Set labels and title
        #ax1.set_xlabel('horizon')
        #ax1.set_ylabel('RMSE')

        # if rho == 28:
        #     ax1.set_ylim((0, 25))
        # else:
        #     ax1.set_ylim((-0.2, 10))
        #     # ax2.set_ylim((-0.001, 0.02))
        #     ax1.set_yticks(np.arange(0.0, 11.0, 1.0))

        #plt.title(f'Training length = {len}')
        plt.legend(title='Noise')
        #plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series 2/rho = {rho}/RMSE, horizon/len={len}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()


def plot_RMSE_vs_horizon_minus_noise(df, rho):
    plt.rc('font', size=14)
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
        df_sub = df[df['training_length'] == len]
        fig, ax1 = plt.subplots()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 2.0, 4.0]:

            color = colors(i)
            marker = markers[i]

            if (len == 25 and rho == 28 and noise == -1.0) or (rho == 20 and noise == -1.0):
                ax2 = ax1.twinx()
                df_noise = df_sub[df_sub['noise'] == noise]
                ax2.plot(df_noise['hor'], df_noise['mean_RMSE']-noise, label=f'{noise}', color=color)
                ax2.scatter(df_noise['hor'], df_noise['mean_RMSE']-noise, color=color, marker=marker, s=50)
                ax2.fill_between(df_noise['hor'], df_noise['min_RMSE']-noise, df_noise['max_RMSE']-noise, alpha=.1, color=color)

                ax2.plot(df_noise['hor'], df_noise['max_RMSE']-noise, linestyle="--", dashes=(5, 5), color=color, alpha=.7)
                ax2.plot(df_noise['hor'], df_noise['min_RMSE']-noise, linestyle="--", dashes=(5, 5), color=color, alpha=.7)
                ax2.tick_params(axis='y', colors=color)

            else:
                df_noise = df_sub[df_sub['noise'] == noise]
                ax1.plot(df_noise['hor'], df_noise['mean_RMSE']-noise, label=f'{noise}', color=color)
                ax1.scatter(df_noise['hor'], df_noise['mean_RMSE']-noise, color=color, marker=marker, s=50)

                ax1.fill_between(df_noise['hor'], df_noise['min_RMSE']-noise, df_noise['max_RMSE']-noise, alpha=.1, color=color, zorder=1)
                ax1.plot(df_noise['hor'], df_noise['max_RMSE']-noise, linestyle="--", dashes=(5,5), color=color, alpha=.5)
                ax1.plot(df_noise['hor'], df_noise['min_RMSE']-noise, linestyle="--", dashes=(5,5), color=color, alpha=.5)

            i += 1

        # Set labels and title
        #ax1.set_xlabel('horizon')
        #ax1.set_ylabel('RMSE')

        if rho == 28:
            ax1.set_ylim((0, 22))
        else:
            ax1.set_ylim((-0.1, 5))
            #ax2.set_ylim((-0.001, 0.02))
            #ax2.set_yticks(np.arange(0.0, 0.0201, 0.005))

        #plt.title(f'Training length = {len}')
        #plt.legend(title='Noise')
        #plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/RMSE, horizon minus noise/len={len}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()


def plot_RMSE_vs_horizon_zoom(df, rho):
    plt.rc('font', size=20)
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
        df_sub = df[df['training_length'] == len]
        fig, ax1 = plt.subplots()

        # Iterate over groups and plot RMSE
        for noise in [0.0]:

            color = colors(0)
            marker = markers[0]

            df_noise = df_sub[df_sub['noise'] == noise]
            ax1.plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color)
            ax1.scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=50)

            ax1.fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color, zorder=1)
            ax1.plot(df_noise['hor'], df_noise['max_RMSE'], linestyle="--", dashes=(5,5), color=color, alpha=.5)
            ax1.plot(df_noise['hor'], df_noise['min_RMSE'], linestyle="--", dashes=(5,5), color=color, alpha=.5)

        #plt.title(f'Training length = {len}')
        #plt.legend(title='Noise')
        plt.locator_params(axis='y', nbins=4)
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/RMSE, horizon/zoom, len={len}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()


# def plot_corr_vs_horizon(df, rho):
#     colors = plt.get_cmap('tab10')
#
#     for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
#         df_sub = df[df['training_length'] == len]
#         plt.figure()
#
#         # Iterate over groups and plot corr
#         i = 0
#         for noise in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
#             color = colors(i)
#             df_noise = df_sub[df_sub['noise'] == noise]
#             plt.plot(df_noise['hor'], df_noise['mean_corr'], label=f'{noise}', color=color)
#             plt.scatter(df_noise['hor'], df_noise['mean_corr'], color=color)
#             plt.plot(df_noise['hor'], df_noise['min_corr'], color=color, linestyle="--")
#             plt.plot(df_noise['hor'], df_noise['max_corr'], color=color, linestyle="--")
#             i += 1
#
#         # Set labels and title
#         plt.xlabel('horizon')
#         plt.ylabel('corr')
#         plt.title(f'len = {len}')
#         plt.legend(title='Noise')
#         plt.tight_layout()
#
#         # Show or save the plot
#         path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/corr, horizon/len={len}.png"
#         plt.savefig(path)
#
#         #plt.savefig(file_path)
#         time.sleep(2)
#         plt.close()

def plot_RMSE_vs_noise(df, rho):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v', 'p']

    for hor in range(1, 11):
        df_sub = df[df['hor'] == hor]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for len in [25, 50, 75, 100]:
            color = colors(i)
            marker = markers[i]
            df_len = df_sub[df_sub['training_length'] == len]
            plt.plot(df_len['noise'], df_len['mean_RMSE'], label=f'{len}', color=color)
            plt.scatter(df_len['noise'], df_len['mean_RMSE'], color=color, marker=marker)
            plt.plot(df_len['noise'], df_len['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_len['noise'], df_len['max_RMSE'], color=color, linestyle="--")
            plt.fill_between(df_len['noise'], df_len['min_RMSE'], df_len['max_RMSE'], alpha=.1, color=color)
            i += 1

        # Set labels and title
        plt.xlabel('Noise')
        plt.ylabel('RMSE')
        plt.title(f'Horizon = {hor}')
        plt.legend(title='Training length')
        plt.tight_layout()

        # Show or save the plot
        path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/RMSE, noise/hor={hor}.png"
        plt.savefig(path)

        #plt.savefig(file_path)
        time.sleep(2)
        plt.close()

# def plot_corr_vs_noise(df, rho):
#     colors = plt.get_cmap('tab10')
#
#     for hor in range(1, 11):
#         df_sub = df[df['hor'] == hor]
#         plt.figure()
#
#         # Iterate over groups and plot RMSE
#         i = 0
#         for len in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
#             color = colors(i)
#             df_len = df_sub[df_sub['training_length'] == len]
#             plt.plot(df_len['noise'], df_len['mean_corr'], label=f'{len}', color=color)
#             plt.scatter(df_len['noise'], df_len['mean_corr'], color=color)
#             plt.plot(df_len['noise'], df_len['min_corr'], color=color, linestyle="--")
#             plt.plot(df_len['noise'], df_len['max_corr'], color=color, linestyle="--")
#             i += 1
#
#         # Set labels and title
#         plt.xlabel('Noise')
#         plt.ylabel('corr')
#         plt.title(f'hor = {hor}')
#         plt.legend(title='Training length')
#         plt.tight_layout()
#
#         # Show or save the plot
#         path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/corr, noise/hor={hor}.png"
#         plt.savefig(path)
#
#         #plt.savefig(file_path)
#         time.sleep(2)
#         plt.close()

def plot_RMSE_vs_length(df, rho):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v', 'p', '^', 'X']

    for hor in range(1, 11):
        df_sub = df[df['hor'] == hor]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
            color = colors(i)
            marker = markers[i]
            df_len = df_sub[df_sub['noise'] == noise]
            plt.plot(df_len['training_length'], df_len['mean_RMSE'], label=f'{noise}', color=color)
            plt.scatter(df_len['training_length'], df_len['mean_RMSE'], color=color)
            plt.fill_between(df_len['training_length'], df_len['min_RMSE'], df_len['max_RMSE'], color=color, alpha=.1)
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

# def plot_corr_vs_length(df, rho):
#     colors = plt.get_cmap('tab10')
#
#     for hor in range(1, 11):
#         df_sub = df[df['hor'] == hor]
#         plt.figure()
#
#         # Iterate over groups and plot RMSE
#         i = 0
#         for noise in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
#             color = colors(i)
#             df_len = df_sub[df_sub['noise'] == noise]
#             plt.plot(df_len['training_length'], df_len['mean_corr'], label=f'{noise}', color=color)
#             plt.scatter(df_len['training_length'], df_len['mean_corr'], color=color)
#             plt.plot(df_len['training_length'], df_len['min_corr'], color=color, linestyle="--")
#             plt.plot(df_len['training_length'], df_len['max_corr'], color=color, linestyle="--")
#             i += 1
#
#         # Set labels and title
#         plt.xlabel('Training length')
#         plt.ylabel('corr')
#         plt.title(f'hor = {hor}')
#         plt.legend(title='Noise')
#         plt.tight_layout()
#
#         # Show or save the plot
#         path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/corr, length/hor={hor}.png"
#         plt.savefig(path)
#
#         #plt.savefig(file_path)
#         time.sleep(2)
#         plt.close()

def plot_RMSE_vs_interval(df, rho):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for len in df['training_length'].unique():
        for hor in df['hor'].unique():
            df_sub = df[(df['training_length'] == len) & (df['hor'] == hor)]
            plt.figure()

            # Iterate over groups and plot RMSE
            i = 0
            for noise in [0.0, 2.0, 4.0]:
                color = colors(i)
                marker = markers[i]
                df_noise = df_sub[df_sub['noise'] == noise]
                plt.plot(df_noise['sampling_interval'], df_noise['mean_RMSE'], label=f'{noise}', color=color)
                plt.scatter(df_noise['sampling_interval'], df_noise['mean_RMSE'], color=color)
                plt.fill_between(df_noise['sampling_interval'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
                #plt.plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linestyle="--")
                #plt.plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linestyle="--")
                i += 1

            # Set labels and title
            plt.xlabel('sampling interval')
            plt.ylabel('RMSE')
            plt.title(f'Training length = {len}')
            plt.legend(title='Noise')
            plt.tight_layout()

            # Show or save the plot
            path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/sampling interval/rho = {rho}/Figures/len={len}, hor={hor}.png"
            plt.savefig(path)

            #plt.savefig(file_path)
            time.sleep(2)
            plt.close()\


def plot_4(df_28, df_20):

    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']
    plt.rc('font', size=16)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    plt.subplots_adjust(hspace=0.3)

    # First plot (Rho = 28, Length = 25)
    df = df_28
    df_sub = df[df['training_length'] == 25]

    i = 0
    for noise in [0.0, 2.0, 4.0]:
        color = colors(i)
        marker = markers[i]
        df_noise = df_sub[df_sub['noise'] == noise]

        if noise == 0.0:
            ax1 = axs[0,0].twinx()
            ax1.fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
            ax1.plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
            ax1.plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
            ax1.plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
            ax1.scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
            ax1.set_ylim(-10, max(df_sub[df_sub['noise'] == 0.0]['mean_RMSE']) * 1.1)
            ax1.tick_params(axis='y', colors=color)
        else:
            axs[0,0].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
            axs[0,0].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
            axs[0,0].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
            axs[0, 0].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
            axs[0, 0].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
        i += 1

    #axs[0,0].set_ylabel('RMSE')

    # Third plot (Rho = 28, Length = 25)
    df_sub = df[df['training_length'] == 75]

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


    # 2nd plot (Rho = 20, Length = 25)
    df = df_20
    df_sub = df[df['training_length'] == 25]

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

    # axs[1, 0].set_xlabel('horizon')
    # axs[1, 0].set_ylabel('RMSE')

    # 4th plot (Rho = 20, Length = 75)
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

    #axs[1, 1].set_xlabel('horizon')
    #axs[1, 1].set_ylabel('RMSE')


    # Add (a) - (f)
    axs[0, 0].text(0.5, -0.12, '(a)', transform=axs[0, 0].transAxes)
    axs[0, 1].text(0.5, -0.12, '(b)', transform=axs[0, 1].transAxes)
    axs[1, 0].text(0.5, -0.22, '(d)', transform=axs[1, 0].transAxes)
    axs[1, 1].text(0.5, -0.22, '(e)', transform=axs[1, 1].transAxes)

    axs[0,0].set_xticklabels([])
    axs[0,1].set_xticklabels([])

    axs[1,0].set_yticklabels([])
    axs[1,1].set_yticklabels([])

    # axs[0, 0].text(-0.3, 0.37, r'$\rho = 28$', transform=axs[0, 0].transAxes, fontsize=16, rotation=90)
    # axs[1, 0].text(-0.3, 0.37, r'$\rho = 20$', transform=axs[1, 0].transAxes, fontsize=16, rotation=90)

    # axs[0, 0].text(1.0, 1.3, r'Training length', transform=axs[0, 0].transAxes, fontsize=18)
    # axs[0, 0].text(0.5, 1.1, r'$25$', transform=axs[0, 0].transAxes, fontsize=16)
    # axs[0, 1].text(0.5, 1.1, r'$75$', transform=axs[0, 1].transAxes, fontsize=16)

    #fig.tight_layout(pad=4.3, h_pad=3, w_pad=0)

    # blue_patch = mlines.Line2D([], [], color=colors(0), marker=markers[0], linestyle='None',
    #                            markersize=10, label='0.0')
    # orange_patch = mlines.Line2D([], [], color=colors(1), marker=markers[1], linestyle='None',
    #                            markersize=10, label='2.0')
    # green_patch = mlines.Line2D([], [], color=colors(2), marker=markers[2], linestyle='None',
    #                            markersize=10, label='4.0')
    #
    # fig.legend(title=r'$\sigma_{noise}$', handles = [blue_patch, orange_patch, green_patch], ncol=3,
    #            loc='lower center', bbox_to_anchor = (0.55,-0.05))

    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/4 plots.png"
    plt.savefig(path, dpi = 300)
    plt.show()


def make_pdf(rho):
    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}"
    folders = [path+'/RMSE, horizon', path+'/RMSE, length', path+'/RMSE, noise']

    # Calculate number of pages needed for each folder
    num_pages_per_folder = [len(os.listdir(folder)) / 12 + 1 for folder in folders]
    num_pages_per_folder[0] -= 1

    # Initialize matplotlib figure
    figsize = (8.5, 11)  # A4 paper size
    pdf_pages = PdfPages(path+"/Figures for Appendix.pdf")

    for folder, num_pages in zip(folders, num_pages_per_folder):
        for page_num in range(int(num_pages)):
            # Initialize matplotlib figure for each page
            fig, axs = plt.subplots(4, 3, figsize=(17, 22))
            fig.subplots_adjust(wspace=0.0, hspace=0.0)

            # Calculate start and end index for files to display on this page
            start_index = page_num * 12
            end_index = min((page_num + 1) * 12, len(os.listdir(folder)))

            # Display each file on this page
            for i in range(4):
                for j in range(3):
                    img_index = start_index + i * 3 + j
                    if img_index < end_index:
                        img_path = os.path.join(folder, os.listdir(folder)[img_index])
                        img = plt.imread(img_path)
                        axs[i, j].imshow(img)
                    axs[i, j].axis('off')

            # Save the figure to the PDF
            pdf_pages.savefig(fig, dpi=300)
            plt.close(fig)

    # Close the PDF
    pdf_pages.close()



if __name__ == '__main__':

    #Load CSV file into a pandas DataFrame
    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series 2/rho = 28/mean and percentiles.csv"
    df_28 = pd.read_csv(path)

    # plot_RMSE_vs_noise(df_28, 28)
    plot_RMSE_vs_horizon(df_28, 28)
    # plot_RMSE_vs_length(df_28, 28)

    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series 2/rho = 20/mean and percentiles.csv"
    df_20 = pd.read_csv(path)

    # plot_RMSE_vs_noise(df_20, 20)
    plot_RMSE_vs_horizon(df_20, 20)
    # plot_RMSE_vs_length(df_20,20)

    # plot_4(df_28, df_20)

    # make_pdf(28)
    # make_pdf(20)

    # path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/sampling interval/rho = 20/mean and percentiles.csv"
    # df = pd.read_csv(path)
    # plot_RMSE_vs_interval(df, 20)









