import math
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.colors as col
from matplotlib.lines import Line2D

def extract_lengths(filename):

    # Extract original length from filename (first two digits)
    vs_index = filename.find('vs')
    comma_index = filename.find(',')
    equal_index = filename.find('=')

    original_length = int(filename[:(vs_index-1)])
    replicates_length = int(filename[vs_index+3 : comma_index])
    rho = int(filename[equal_index + 2:equal_index+4])

    return original_length, replicates_length, rho

def from_csv_to_df(path, filename, test):

    if filename.endswith(".csv"):
        # Read the CSV file into a DataFrame
        file_path = path + "/" + filename
        df_file = pd.read_csv(file_path)

        # Extract lengths from filename
        original_length, replicates_length, rho = extract_lengths(filename)

        # Add columns
        df_file['Test'] = test
        df_file['Rho'] = rho
        df_file['Original_Length'] = original_length
        df_file['Replicates_Length'] = replicates_length
        df_file['Factor'] = int(original_length / replicates_length)

        # Transform none count to none percentage
        df_file['none_count'] = (
                df_file['none_count'] / 100.0)

    return df_file

def combine_csv_files(folder_path):

    # Initialize an empty DataFrame to store combined data
    df = pd.DataFrame()

    # Iterate through each file in the folder
    path_1 = folder_path + "/begin conditions"
    path_2 = folder_path + "/rho"

    for filename in os.listdir(path_1):
        df_file = from_csv_to_df(path_1, filename, 'begin_conditions')
        df = pd.concat([df, df_file], ignore_index=True)

    for filename in os.listdir(path_2):
        df_file = from_csv_to_df(path_2, filename, 'rho')
        df = pd.concat([df, df_file], ignore_index=True)

    return df

def make_big_box_plot(df):
    tab20_colors = plt.cm.get_cmap('tab10', 10)
    c1 = col.to_rgb(tab20_colors(0))
    c2 = col.to_rgb(tab20_colors(1))
    c3 = col.to_rgb(tab20_colors(2))
    c4 = col.to_rgb(tab20_colors(3))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    plt.rc('font', size=15)

    for rho_index in [0]:
        test = 'begin_conditions'
        var = 1.0
        rho = [28, 20][rho_index]

        df_test = df[(df['Test'] == test) & (df['variance'] == var) & (df['Rho'] == rho)]
        boxplot_stats = []
        none_counts = []

        for noise_level in [0.0, 1.0, 2.0]:

            df_noise = df_test[df_test['noise'] == noise_level]

            for original_length, replicates_length in [(192, 96), (96, 48), (48, 24), (24, 12)]:
                df_sub = df_noise[(df_noise['Original_Length'] == original_length) & (df_noise['Replicates_Length'] == replicates_length)]
                df_sub = df_sub.iloc[:, 2:]

                # Extract boxplot statistics
                med = df_sub['median'].values
                q1 = df_sub['q1'].values
                q3 = df_sub['q3'].values
                whislo = df_sub['q1'].values
                whishi = df_sub['q3'].values
                fliers = []

                if math.isnan(df_sub['none_count'].values[0]):
                    df_sub['none_count'].values[0] = 1.0

                # Append boxplot statistics to list
                boxplot_stats.append({'med': med, 'q1': q1, 'q3': q3, 'whislo': whislo, 'whishi': whishi, 'fliers': fliers})
                none_counts.append(1.0 - df_sub['none_count'].values[0])

        # Plot single boxplot with multiple boxes
        axes[rho_index].bxp([boxplot_stats[i] for i in [0, 4, 8]], positions=[1, 3.5, 6], widths=0.5, showmeans=False,
                            patch_artist=True,
                            boxprops=dict(facecolor=c1, alpha=.8),
                            whiskerprops=dict(color=c1, alpha=.5),
                            medianprops=dict(color='black', linewidth=2))
        axes[rho_index].bxp([boxplot_stats[i] for i in [1, 5, 9]], positions=[1.5, 4, 6.5], widths=0.5, showmeans=False,
                            patch_artist=True, whiskerprops={},
                            boxprops=dict(facecolor=c2, alpha=.8),
                            medianprops=dict(color='black', linewidth=2))
        axes[rho_index].bxp([boxplot_stats[i] for i in [2, 6, 10]], positions=[2, 4.5, 7], widths=0.5, showmeans=False,
                            patch_artist=True, whiskerprops={},
                            boxprops=dict(facecolor=c3, alpha=.8),
                            medianprops=dict(color='black', linewidth=2))
        axes[rho_index].bxp([boxplot_stats[i] for i in [3, 7, 11]], positions=[2.5, 5, 7.5], widths=0.5, showmeans=False,
                            patch_artist=True,
                            whiskerprops={},
                            boxprops=dict(facecolor=c4, alpha=.8),
                            medianprops=dict(color='black', linewidth=2))

        # # Plot scatter points for none_count on the second y-axis
        ax2 = axes[rho_index].twinx()
        color_list = [c1, c2, c3, c4, c1, c2, c3, c4, c1, c2, c3, c4,]
        ax2.scatter([1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5], none_counts, color=color_list, marker='o', s=150, edgecolor='black')
        ax2.set_ylabel('Fraction of iterations')
        ax2.set_ylim(-0.05, 1.1)

        axes[rho_index].set_ylim(0, 6)
        axes[rho_index].set_ylabel('# Replicates needed', fontsize=15)

        # Indicate noise levels on x-axis
        if rho_index == 1:
            axes[rho_index].set_xticks([1.75, 4.25, 6.75])
            axes[rho_index].set_xticklabels(['0.0', '1.0', '2.0'], fontsize = 15)
        else:
            axes[rho_index].set_xticks([1.75, 4.25, 6.75])
            axes[rho_index].set_xticklabels(['', '', ''], fontsize = 15)

        axes[rho_index].set_yticks([0, 1, 2, 3, 4, 5, 6])
        axes[rho_index].set_yticklabels(['0', '1', '2', '3', '4', '5', '6'], fontsize=15)

        # Create legend
        lab1 = r"192 $\rightarrow$ 96"
        lab2 = r"96 $\rightarrow$ 48"
        lab3 = r"48 $\rightarrow$ 24"
        lab4 = r"24 $\rightarrow$ 12"

        custom_lines = [
                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c1,
                       markeredgecolor='black'),
                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c2,
                       markeredgecolor='black'),
                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c3,
                       markeredgecolor='black'),
                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c4,
                       markeredgecolor='black')]

        if rho_index == 1:
            box = axes[rho_index].get_position()
            axes[rho_index].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.95])
            axes[rho_index].legend(custom_lines, [lab1, lab2, lab3, lab4],
                                       loc='upper center', bbox_to_anchor=(0.5, -0.1),
                                       fancybox=True, shadow=True, ncol=5)



    plt.tight_layout(pad=3)
    #path_name = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 10/Figures/"
    path_name = "C:/Users/fleur/Documents/Resultaten/length vs replicates/Figures/"
    plt.savefig(path_name + f"{test}.png", dpi=300)
    plt.show()

def make_small_box_plot(df):

    tab20_colors = plt.cm.get_cmap('tab10', 10)
    c1 = col.to_rgb(tab20_colors(0))
    c2 = col.to_rgb(tab20_colors(1))
    c3 = col.to_rgb(tab20_colors(2))
    c4 = col.to_rgb(tab20_colors(3))

    #tab20b_colors = plt.cm.get_cmap('tab20b', 20)
    #c5 = col.to_rgb(tab20b_colors(17))

    for rho in [28]:
        for test, variance_list in [('begin_conditions', [1.0, 5.5, 10.0]), ('rho', [1.0, 3.0, 5.0])]:
            for var in variance_list:
                df_test = df[(df['Test'] == test) & (df['variance'] == var) & (df['Rho'] == rho)]
                fig, ax = plt.subplots(figsize=(10, 5))
                ax2 = ax.twinx()
                boxplot_stats = []
                none_counts = []

                for noise_level in [0.0, 1.0, 2.0]:
                    df_noise = df_test[df_test['noise'] == noise_level]

                    for original_length, replicates_length in [(192, 96), (96, 48), (48, 24), (24, 12)]:
                        df_sub = df_noise[(df_noise['Original_Length'] == original_length) & (df_noise['Replicates_Length'] == replicates_length)]
                        df_sub = df_sub.iloc[:, 2:]

                        # Extract boxplot statistics
                        med = df_sub['median'].values
                        q1 = df_sub['q1'].values
                        q3 = df_sub['q3'].values
                        whislo = df_sub['q1'].values
                        whishi = df_sub['q3'].values
                        fliers = []

                        # Append boxplot statistics to list
                        boxplot_stats.append({'med': med, 'q1': q1, 'q3': q3, 'whislo': whislo, 'whishi': whishi, 'fliers': fliers})
                        none_counts.append(1.0 - df_sub['none_count'].values[0])

                # Plot single boxplot with multiple boxes
                ax.bxp([boxplot_stats[i] for i in [0, 4, 8]], positions=[1, 3.5, 6], widths=0.5, showmeans=False, patch_artist=True,
                       boxprops=dict(facecolor=c1, alpha=.8),
                       whiskerprops=dict(color=c1, alpha=.5),
                       medianprops=dict(color='black', linewidth=2))
                ax.bxp([boxplot_stats[i] for i in [1, 5, 9]], positions=[1.5, 4, 6.5], widths=0.5, showmeans=False, patch_artist=True, whiskerprops={},
                        boxprops=dict(facecolor=c2, alpha=.8),
                        medianprops=dict(color='black', linewidth=2))
                ax.bxp([boxplot_stats[i] for i in [2, 6, 10]], positions=[2, 4.5, 7], widths=0.5, showmeans=False, patch_artist=True, whiskerprops={},
                        boxprops=dict(facecolor=c3, alpha=.8),
                        medianprops=dict(color='black', linewidth=2))
                ax.bxp([boxplot_stats[i] for i in [3, 7, 11]], positions=[2.5, 5, 7.5], widths=0.5, showmeans=False, patch_artist=True,
                       whiskerprops={},
                       boxprops=dict(facecolor=c4, alpha = .8),
                       medianprops=dict(color='black', linewidth=2))

                # Plot scatter points for none_count on the second y-axis
                color_list = [c1, c2, c3, c4, c1, c2, c3, c4, c1, c2, c3, c4]
                ax2.scatter([1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5], none_counts, color=color_list, marker='o', s=50, edgecolor='black')
                ax2.set_ylabel('Fraction of iterations for which RMSE improved')
                ax2.tick_params(axis='y')
                ax2.set_ylim(0, 1.1)
                ax.set_ylabel('#replicates needed to improve RMSE')

                # Set title for the entire figure
                if test == "begin_conditions":
                    plt.title(r"$\rho_{Lorenz} = $" + str(rho) + "\n" + r"$\sigma_{begin \ conditions} = $" + str(var), fontsize=16)
                else:
                    plt.title(r"$\rho_{Lorenz} = $" + str(rho) + "\n" + r"$\sigma_{\rho} = $" + str(var), fontsize=16)

                ax.set_xticks([1.75, 4.25, 6.75])
                ax.set_xticklabels([r'$\sigma_{noise}$ = 0.0', r'$\sigma_{noise}$ = 1.0', r'$\sigma_{noise}$ = 2.0'])
                ax.set_yticks(range(0, 5))

                # Create legend
                lab1 = r"192 $\rightarrow$ 96"
                lab2 = r"96 $\rightarrow$ 48"
                lab3 = r"48 $\rightarrow$ 24"
                lab4 = r"24 $\rightarrow$ 12"

                custom_lines = [Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c1, markeredgecolor='black'),
                                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c2, markeredgecolor='black'),
                                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c3, markeredgecolor='black'),
                                Line2D([0], [0], marker='s', markersize=15, color='white', markerfacecolor=c4, markeredgecolor='black')]

                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])

                # Put a legend below current axis
                ax.legend(custom_lines, [lab1, lab2, lab3, lab4],
                          loc='upper center', bbox_to_anchor=(0.5, -0.1),
                          fancybox=True, shadow=True, ncol=5)

                #plt.tight_layout()
                #path_name = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 10/Figures/"
                path_name = "C:/Users/fleur/Documents/Resultaten/length vs replicates/Figures/"
                plt.savefig(path_name + f"{test}, variance = {var}.png")
                plt.show()


if __name__ == "__main__":

    # Specify the folder path containing CSV files
    #folder_path = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 10'
    folder_path = 'C:/Users/fleur/Documents/Resultaten/length vs replicates'

    # Call the function to combine CSV files
    combined_data = combine_csv_files(folder_path)

    # Display the combined DataFrame
    # make_big_box_plot(combined_data)

    make_small_box_plot(combined_data)
