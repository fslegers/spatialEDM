import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
import matplotlib.colors as col
from matplotlib.lines import Line2D

def from_csv_to_df(path, filename):

    if filename.endswith(".csv"):
        # Read the CSV file into a DataFrame
        file_path = path + "/" + filename
        df_file = pd.read_csv(file_path)

    return df_file

def combine_csv_files(path):

    # Initialize an empty DataFrame to store combined data
    df = pd.DataFrame()

    for filename in os.listdir(path):
        df_file = from_csv_to_df(path, filename)
        df = pd.concat([df, df_file], ignore_index=True)

    return df

def make_figures(df_og):

    tab20_colors = plt.cm.get_cmap('tab10', 10)
    markers = ['s', 'o', 'p']
    plt.rc('font', size=14)

    fig, axs = plt.subplots(1, 2, figsize = (10,4))
    plt.subplots_adjust(hspace=15.0, wspace=1.0)

    length = 25
    df = df_og[df_og['length'] == length]
    counter = 0
    for noise in [0.0, 2.0, 4.0]:
        df_noise = df[(df['noise'] == noise)]

        # Plot RMSE
        axs[0].plot(df_noise['variance'], df_noise['mean_RMSE'], color=col.to_rgb(tab20_colors(counter)), label=noise)
        axs[0].fill_between(df_noise['variance'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=col.to_rgb(tab20_colors(counter)))
        axs[0].plot(df_noise['variance'], df_noise['min_RMSE'], color=col.to_rgb(tab20_colors(counter)), linewidth = .5, alpha=.5)
        axs[0].plot(df_noise['variance'], df_noise['max_RMSE'], color=col.to_rgb(tab20_colors(counter)), linewidth = .5, alpha=.5)
        axs[0].scatter(df_noise['variance'], df_noise['mean_RMSE'], color=col.to_rgb(tab20_colors(counter)), marker=markers[counter])
        counter += 1

    axs[0].set_xlabel('variance among replicates')
    axs[0].set_ylabel('RMSE')
    #axs[0].legend(title='noise', bbox_to_anchor=(1, 0.1))

    length = 75
    df = df_og[df_og['length'] == length]
    counter = 0
    for noise in [0.0, 2.0, 4.0]:
        df_noise = df[(df['noise'] == noise)]

        # Plot RMSE
        axs[1].plot(df_noise['variance'], df_noise['mean_RMSE'], color=col.to_rgb(tab20_colors(counter)), label=noise)
        axs[1].fill_between(df_noise['variance'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1,
                            color=col.to_rgb(tab20_colors(counter)))
        axs[1].plot(df_noise['variance'], df_noise['min_RMSE'], color=col.to_rgb(tab20_colors(counter)), linewidth=.5,
                    alpha=.5)
        axs[1].plot(df_noise['variance'], df_noise['max_RMSE'], color=col.to_rgb(tab20_colors(counter)), linewidth=.5,
                    alpha=.5)
        axs[1].scatter(df_noise['variance'], df_noise['mean_RMSE'], color=col.to_rgb(tab20_colors(counter)),
                       marker=markers[counter])
        counter += 1

    axs[1].set_xlabel('variance among replicates')
    axs[1].set_ylabel('RMSE')
    axs[1].legend(title='noise', bbox_to_anchor=(1.0, 0.8))

    axs[0].text(0.5, -0.3, '(a)', transform=axs[0].transAxes)
    axs[1].text(0.5, -0.3, '(b)', transform=axs[1].transAxes)

    plt.tight_layout()
    plt.show()


        # # plot dim
        # counter = 0
        # for noise in [0.0, 2.0, 4.0]:
        #     df_noise = df[(df['noise'] == noise)]
        #
        #     plt.plot(df_noise['variance'], df_noise['mean_dim'], color=col.to_rgb(tab20_colors(counter)), label=noise)
        #     plt.plot(df_noise['variance'], df_noise['min_dim'], color=col.to_rgb(tab20_colors(counter)),
        #              linestyle='--')
        #     plt.plot(df_noise['variance'], df_noise['max_dim'], color=col.to_rgb(tab20_colors(counter)),
        #              linestyle='--')
        #     plt.scatter(df_noise['variance'], df_noise['mean_dim'], color=col.to_rgb(tab20_colors(counter)), marker=markers[counter])
        #     counter += 1
        #
        # plt.xlabel('variance among replicates')
        # plt.ylabel('optimal dimension')
        # plt.show()
        #
        # # plot theta
        # counter = 0
        # for noise in [0.0, 2.0, 4.0]:
        #     df_noise = df[(df['noise'] == noise)]
        #
        #     plt.plot(df_noise['variance'], df_noise['theta'], color=col.to_rgb(tab20_colors(counter)), label=noise)
        #     plt.plot(df_noise['variance'], df_noise['min_theta'], color=col.to_rgb(tab20_colors(counter)),
        #              linestyle='--')
        #     plt.plot(df_noise['variance'], df_noise['max_theta'], color=col.to_rgb(tab20_colors(counter)),
        #              linestyle='--')
        #     plt.scatter(df_noise['variance'], df_noise['theta'], color=col.to_rgb(tab20_colors(counter)), marker=markers[counter])
        #     counter += 1
        #
        # plt.xlabel('variance among replicates')
        # plt.ylabel('optimal theta')
        # plt.show()


def calculate_summary(combined_data):
    df = []

    for noise in [0.0, 2.0, 4.0]:
        for variance in np.arange(0, 11, 1):
            for length in [25, 50, 75, 100]:
                df_noise = combined_data[(combined_data['noise'] == noise) & (combined_data['variance'] == variance) & (combined_data['length'] == length)]

                mean_RMSE = np.mean(df_noise['RMSE'])
                mean_dim = np.median(df_noise['dim'])
                mean_theta = np.median(df_noise['theta'])

                min_RMSE = np.percentile(df_noise['RMSE'], 10)
                max_RMSE = np.percentile(df_noise['RMSE'], 90)
                min_dim = np.percentile(df_noise['dim'], 10)
                max_dim = np.percentile(df_noise['dim'], 90)
                min_theta = np.percentile(df_noise['theta'], 10)
                max_theta = np.percentile(df_noise['theta'], 90)


                row = {'noise': noise, 'variance': variance, 'length': length,
                       'min_RMSE': min_RMSE,
                       'max_RMSE': max_RMSE,
                       'mean_RMSE': mean_RMSE,
                       'min_dim': min_dim,
                       'max_dim': max_dim,
                       'mean_dim': mean_dim,
                       'min_theta': min_theta,
                       'max_theta': max_theta,
                       'mean_theta': mean_theta,
                       'theta': mean_theta}
                df.append(row)

    return pd.DataFrame(df)

if __name__ == "__main__":

    # Specify the folder path containing CSV files
    folder_path = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/hyperparameters for replicates/rho = 28/begin_conditions'

    # Call the function to combine CSV files
    combined_data = combine_csv_files(folder_path)

    # Calculate means and medians
    combined_data = calculate_summary(combined_data)

    # Display the combined DataFrame
    make_figures(combined_data)