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
                df_file['none_count'] / 50.0)

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

def make_box_plot(df):

    tab20_colors = plt.cm.get_cmap('tab20', 20)
    c1 = col.to_rgb(tab20_colors(0))
    c2 = col.to_rgb(tab20_colors(8))
    c3 = col.to_rgb(tab20_colors(16))

    for test in ['begin_conditions', 'rho']:
        df_test = df[df['Test'] == test]

        for original_length, replicates_length in [(48, 24), (48, 12), (96, 48), (96, 24), (96, 12)]:

            # Create a single subplot for the entire figure
            fig, ax = plt.subplots(figsize=(18, 4))
            none_counts = []

            # Collect boxplot statistics for all noise and variance pairs
            boxplot_stats = []

            df_sub = df_test[
                (df_test['Original_Length'] == original_length) & (df_test['Replicates_Length'] == replicates_length)]

            if test == "begin_conditions":
                variances = [1.0, 5.5, 10.0]
            else:
                variances = [1.0, 3.0, 5.0]

            for i, pair in enumerate(product([0.0, 1.0, 2.0], variances)):
                noise = pair[0]
                variance = pair[1]

                # Get boxplot values for this noise, variance pair
                boxplot_values = df_sub[(df_sub['noise'] == noise) & (df_sub['variance'] == variance)]
                boxplot_values = boxplot_values.iloc[:, 2:]

                # Extract boxplot statistics
                med = boxplot_values['median'].values
                q1 = boxplot_values['q1'].values
                q3 = boxplot_values['q3'].values
                whislo = boxplot_values['q1'].values
                whishi = boxplot_values['q3'].values
                fliers = []

                nones = boxplot_values['none_count'].values[0]
                none_counts.append(nones)

                # Append boxplot statistics to list
                boxplot_stats.append(
                    {'med': med, 'q1': q1, 'q3': q3, 'whislo': whislo, 'whishi': whishi, 'fliers': fliers})

            # Plot single boxplot with multiple boxes
            ax.bxp(boxplot_stats[0:3], positions=[1, 2, 3], showmeans=False, patch_artist=True, whiskerprops={},
                   boxprops=dict(facecolor=c1))
            ax.bxp(boxplot_stats[3:6], positions=[4, 5, 6], showmeans=False, patch_artist=True, whiskerprops={},
                   boxprops=dict(facecolor=c2))
            ax.bxp(boxplot_stats[6:9], positions=[7, 8, 9], showmeans=False, patch_artist=True, whiskerprops={},
                   boxprops=dict(facecolor=c3))

            # Set title for the entire figure
            plt.title(f"{test}, \n {original_length} vs {replicates_length}", fontsize=20)

            ax.set_xticks(np.arange(1, 10))

            labels = [f"Noise: {noise}, \n Variance: {variance}" for noise, variance in product([0.0, 1.0, 2.0], [1.0, 5.5, 10.0])]
            new_labels = []
            for i in range(len(labels)):
                label = labels[i]
                new_label = label + "\n \n #repl > 32: \n " + f"{none_counts[i] * 100:.1f}%"
                new_labels.append(new_label)
            ax.set_xticklabels(new_labels)

            plt.tight_layout()
            plt.show()

def make_pretty_box_plot(df):
    rho = np.unique(df['Rho'])[0]

    tab20_colors = plt.cm.get_cmap('tab20', 20)
    c1 = col.to_rgb(tab20_colors(2))
    c2 = col.to_rgb(tab20_colors(12))
    c3 = col.to_rgb(tab20_colors(18))
    c4 = col.to_rgb(tab20_colors(16))

    tab20b_colors = plt.cm.get_cmap('tab20b', 20)
    c5 = col.to_rgb(tab20b_colors(17))

    for test, variance_list in [('begin_conditions', [1.0, 5.5, 10.0]), ('rho', [1.0, 3.0, 5.0])]:
        for var in variance_list:
            df_test = df[(df['Test'] == test) & (df['variance'] == var)]
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
            ax2.scatter([1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5], none_counts, color=c5, marker='s', s=150, edgecolor='black')
            ax2.scatter([1, 1.5, 2, 2.5, 3.5, 4, 4.5, 5, 6, 6.5, 7, 7.5], none_counts, color=color_list, marker='o', s=50, edgecolor='black')
            ax2.set_ylabel('Fraction of iterations for which RMSE improved', color=c5)
            ax2.tick_params(axis='y', labelcolor=c5)
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
            path_name = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/Figures/"
            plt.savefig(path_name + f"{test}, variance = {var}.png")
            plt.show()


if __name__ == "__main__":

    # Specify the folder path containing CSV files
    folder_path = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates'

    # Call the function to combine CSV files
    combined_data = combine_csv_files(folder_path)

    # Display the combined DataFrame
    make_pretty_box_plot(combined_data)
