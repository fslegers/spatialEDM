import math
import os
import pandas as pd
import seaborn as sns
import ast
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.colors as col
from matplotlib.lines import Line2D
from brokenaxes import brokenaxes

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

    return df_file

def combine_csv_files(folder_path):

    # Initialize an empty DataFrame to store combined data
    df = pd.DataFrame()

    # Iterate through each file in the folder
    path_1 = folder_path + "/begin conditions"
    path_2 = folder_path + "/rho"
    path_3 = folder_path + "/b"

    for filename in os.listdir(path_1):
        df_file = from_csv_to_df(path_1, filename, 'begin_conditions')
        df = pd.concat([df, df_file], ignore_index=True)

    for filename in os.listdir(path_2):
        df_file = from_csv_to_df(path_2, filename, 'rho')
        df = pd.concat([df, df_file], ignore_index=True)

    for filename in os.listdir(path_3):
        df_file = from_csv_to_df(path_3, filename, 'b')
        df = pd.concat([df, df_file], ignore_index=True)

    return df


def make_violin_plot(df_full):
    plt.rcParams['font.size'] = 20
    #sns.set_style('whitegrid')

    for test in df_full['Test'].unique():
        df = df_full[df_full['Test'] == test]

        for variance, rho in product(df['variance'].unique(), df['Rho'].unique()):

            df_sub = df[(df['variance'] == variance) & (df['Test'] == test) & (df['Rho'] == rho)]

            # Loop through unique lengths
            df_for_plotting = pd.DataFrame()
            df_for_swarmplot = pd.DataFrame()
            labels = []

            for noise in df['noise'].unique():
                df_sub_sub = df_sub[df_sub['noise'] == noise]

                fix, axes = plt.subplots(figsize=(12, 8))

                for length in df_sub_sub['Original_Length'].unique():
                    info = df_sub_sub[df_sub_sub['Original_Length'] == length]['n_replicates']
                    try:
                        info = info[0]
                    except:
                        info = info.values[0]
                    info = ast.literal_eval(info)

                    # Change all Nones to 34
                    info = [34 if x is None else x for x in info]

                    sub_df_for_plotting = pd.DataFrame(info, columns=['n_replicates'])
                    sub_df_for_plotting['length'] = length
                    sub_df_for_plotting['noise'] = noise

                    label = len(sub_df_for_plotting[sub_df_for_plotting['n_replicates'] > 32])
                    label = label/2.5
                    label = f"{label:.1f}%"
                    labels.append(label)

                    sub_df_for_stripplot = sub_df_for_plotting
                    df_for_swarmplot = pd.concat([df_for_swarmplot, sub_df_for_stripplot])

                    sub_df_for_plotting = sub_df_for_plotting[sub_df_for_plotting['n_replicates'] <= 32]
                    df_for_plotting = pd.concat([df_for_plotting, sub_df_for_plotting], axis=0)

            # density_norm = count: the width of a violin is proportional to the number of observations
            sns.violinplot(data=df_for_plotting, x='noise', y='n_replicates', hue='length', palette='tab10', ax=axes,
                           cut=0, zorder=2, density_norm='count', alpha=.75, legend=False,
                           inner_kws=dict(box_width=12, whis_width=4, color='.2'))

            sns.stripplot(data=df_for_swarmplot, x='noise', y='n_replicates', hue='length', palette='tab10', legend=False,
                          dodge=True, jitter=0.15, size=8, alpha=.4, zorder=1)

            axes.set_yticks(list(np.arange(0,31,5)) + [34])
            axes.set_yticklabels([str(i) for i in np.arange(0,31,5)] + ['>32'])

            axes.set_ylabel('')
            axes.set_xlabel('')

            axes.set_ylim((-0.5, 39))

            positions = [0.08, 0.17, 0.26, 0.41, 0.5, 0.59, 0.76, 0.84, 0.92]
            for i in range(len(positions)):
                axes.text(positions[i], 0.95, labels[i],
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=axes.transAxes,
                          fontsize=14,
                          rotation=45)

            #plt.title(f'Variance: {variance}, Test: {test}, Rho: {rho}')
            path = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/violinplots/'
            plt.savefig(path+f'{variance}, {test}, {rho}.png', dpi=200)
            plt.show()

def make_box_plot(df_full):
    plt.rcParams['font.size'] = 20
    #sns.set_style('whitegrid')

    for test in df_full['Test'].unique():
        df = df_full[df_full['Test'] == test]

        for variance, rho in product(df['variance'].unique(), df['Rho'].unique()):

            df_sub = df[(df['variance'] == variance) & (df['Test'] == test) & (df['Rho'] == rho)]

            # Loop through unique lengths
            df_for_plotting = pd.DataFrame()
            df_for_swarmplot = pd.DataFrame()
            labels = []

            for noise in df['noise'].unique():
                df_sub_sub = df_sub[df_sub['noise'] == noise]

                fix, axes = plt.subplots(figsize=(12, 8))

                for length in df_sub_sub['Original_Length'].unique():
                    info = df_sub_sub[df_sub_sub['Original_Length'] == length]['n_replicates']
                    try:
                        info = info[0]
                    except:
                        info = info.values[0]
                    info = ast.literal_eval(info)

                    # Change all Nones to 34
                    info = [34 if x is None else x for x in info]

                    sub_df_for_plotting = pd.DataFrame(info, columns=['n_replicates'])
                    sub_df_for_plotting['length'] = length
                    sub_df_for_plotting['noise'] = noise

                    label = len(sub_df_for_plotting[sub_df_for_plotting['n_replicates'] > 32])
                    label = label/2.5
                    label = f"{label:.1f}%"
                    labels.append(label)

                    sub_df_for_stripplot = sub_df_for_plotting
                    df_for_swarmplot = pd.concat([df_for_swarmplot, sub_df_for_stripplot])

                    sub_df_for_plotting = sub_df_for_plotting[sub_df_for_plotting['n_replicates'] <= 32]
                    df_for_plotting = pd.concat([df_for_plotting, sub_df_for_plotting], axis=0)

            # density_norm = count: the width of a violin is proportional to the number of observations
            sns.boxplot(data=df_for_plotting, x='noise', y='n_replicates', hue='length', palette='tab10', ax=axes,
                           zorder=2, legend=False, showfliers=False, whis=0)

            # axes.set_yticks(list(np.arange(0,31,5)) + [34])
            # axes.set_yticklabels([str(i) for i in np.arange(0,31,5)] + ['>32'])

            axes.set_ylabel('')
            axes.set_xlabel('')

            axes.set_ylim((0, 12))

            # positions = [0.08, 0.17, 0.26, 0.41, 0.5, 0.59, 0.76, 0.84, 0.92]
            # for i in range(len(positions)):
            #     axes.text(positions[i], 0.95, labels[i],
            #               horizontalalignment='center',
            #               verticalalignment='center',
            #               transform=axes.transAxes,
            #               fontsize=14,
            #               rotation=45)

            #plt.title(f'Variance: {variance}, Test: {test}, Rho: {rho}')
            path = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/boxplots/'
            plt.savefig(path+f'{variance}, {test}, {rho}.png', dpi=200)
            plt.show()



if __name__ == "__main__":

    # Specify the folder path containing CSV files
    folder_path = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 5'

    # Call the function to combine CSV files
    combined_data = combine_csv_files(folder_path)

    # Display the combined DataFrame
    make_violin_plot(combined_data)
