import os
import ast
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from itertools import product
import matplotlib.colors as col
from matplotlib.lines import Line2D

def from_csv_to_df(path, filename):

    if filename.endswith(".csv"):
        # Read the CSV file into a DataFrame
        file_path = path + "/" + filename
        dfs = pd.read_csv(file_path)

        dfs['RMSE_list_dim'] = dfs['RMSE_list_dim'].apply(ast.literal_eval)
        dfs['RMSE_list_theta'] = dfs['RMSE_list_theta'].apply(ast.literal_eval)

        df_dims = pd.DataFrame(dfs['RMSE_list_dim'].to_list(), columns=['dim_' + str(i) for i in range(1, len(dfs['RMSE_list_dim'][0]) + 1)])
        df_thetas = pd.DataFrame(dfs['RMSE_list_theta'].to_list(), columns=['theta_' + str(i) for i in range(len(dfs['RMSE_list_theta'][0]))])

        dfs = pd.concat([dfs, df_dims, df_thetas], axis=1)

    return dfs

def combine_csv_files(path):

    # Initialize an empty DataFrame to store combined data
    df = pd.DataFrame()

    for filename in os.listdir(path):
        df_file = from_csv_to_df(path, filename)
        df = pd.concat([df, df_file], ignore_index=True)

    return df

def make_figures(df, rho, n_repl, test):

    tab20_colors = plt.cm.get_cmap('tab10', 10)
    markers = ['s', 'o', 'p']

    path = "C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/hyperparameters for replicates/Figures/"

    for length in [25, 50, 75, 100]:
        for noise in [0.0, 2.0, 4.0]:

            # Plot dimensions
            for var in [0, 3, 6, 9]:
                df_var = df[(df['length'] == length) & (df['noise'] == noise) & (df['variance'] == var)]
                df_var = df_var.dropna(axis=1)

                colnames = ['dim_' + str(i) + '_mean' for i in range(1, 11)]
                colnames = list(set(df_var.columns).intersection(set(colnames)))
                colnames = sorted(colnames)
                means = df_var[colnames].values[0]

                colnames = ['dim_' + str(i) + '_10th' for i in range(1, 11)]
                colnames = list(set(df_var.columns).intersection(set(colnames)))
                colnames = sorted(colnames)
                percentiles_10 = df_var[colnames].values[0]

                colnames = ['dim_' + str(i) + '_90th' for i in range(1, 11)]
                colnames = list(set(df_var.columns).intersection(set(colnames)))
                colnames = sorted(colnames)
                percentiles_90 = df_var[colnames].values[0]

                plt.plot(np.arange(1, len(means) + 1, 1), means, label=var)
                plt.scatter(np.arange(1, len(means) + 1, 1), means)

                #plt.fill_between(np.arange(1, len(means) + 1, 1), percentiles_10, percentiles_90, alpha=.1)

            plt.title(f'rho = {rho}, length = {length}, noise = {noise}')
            plt.legend(title='variance')
            plt.xlabel("dimension")
            plt.ylabel("RMSE")

            file_name = path + f"dim/rho = {rho}, n_repl = {n_repl}, length = {length}, noise = {noise}, test = {test}.png"
            plt.savefig(file_name)
            plt.show()

            # Plot thetas
            for var in [0, 3, 6, 9]:
                df_var = df[(df['length'] == length) & (df['noise'] == noise) & (df['variance'] == var)]
                df_var = df_var.dropna(axis=1)

                colnames = ['theta_' + str(i) + '_mean' for i in range(1, 11)]
                colnames = list(set(df_var.columns).intersection(set(colnames)))
                colnames = sorted(colnames)
                means = df_var[colnames].values[0]

                colnames = ['theta_' + str(i) + '_10th' for i in range(1, 11)]
                colnames = list(set(df_var.columns).intersection(set(colnames)))
                colnames = sorted(colnames)
                percentiles_10 = df_var[colnames].values[0]

                colnames = ['theta_' + str(i) + '_90th' for i in range(1, 11)]
                colnames = list(set(df_var.columns).intersection(set(colnames)))
                colnames = sorted(colnames)
                percentiles_90 = df_var[colnames].values[0]

                plt.plot(np.arange(1, len(means) + 1, 1), means, label=var)
                plt.scatter(np.arange(1, len(means) + 1, 1), means)

                #plt.fill_between(np.arange(1, len(means) + 1, 1), percentiles_10, percentiles_90, alpha=.1)

            plt.title(f'rho = {rho}, length = {length}, noise = {noise}')
            plt.legend(title='variance')
            plt.xlabel("theta")
            plt.ylabel("RMSE")

            file_name = path + f"theta/rho = {rho}, n_repl = {n_repl}, length = {length}, noise = {noise}, test = {test}.png"
            plt.savefig(file_name)
            plt.show()

def calculate_summary(combined_data):
    ultimate_df = pd.DataFrame()

    columns_dim = ['dim_' + str(i) for i in range(1, 11)]
    columns_theta = ['theta_' + str(i) for i in range(0, 11)]
    columns = columns_dim + columns_theta

    for noise in [0.0, 2.0, 4.0]:
        for variance in np.arange(0.0, 12.0, 3.0):
            for length in [25, 50, 75, 100]:

                df_noise = combined_data[(combined_data['noise'] == noise) & (combined_data['variance'] == variance) & (combined_data['length'] == length)]
                percentiles_df = df_noise[columns].describe(percentiles=[0.1, 0.9]).loc[['mean', '10%', '90%']]

                combined_df = pd.DataFrame()
                for column in percentiles_df:
                    column_df = pd.DataFrame({
                                f'{column}_10th': [percentiles_df[column][1]],
                                f'{column}_90th': [percentiles_df[column][2]],
                                f'{column}_mean': [percentiles_df[column][0]]
                            })
                    combined_df = pd.concat([combined_df, column_df], axis=1)

                combined_df['noise'] = noise
                combined_df['variance'] = variance
                combined_df['length'] = length

                ultimate_df = pd.concat([ultimate_df, combined_df], axis=0)

    return ultimate_df

if __name__ == "__main__":

    rho = 20
    test = 'begin_conditions'

    # Specify the folder path containing CSV files
    folder_path = f'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/hyperparameters for replicates/rho = {rho}/{test}'

    # Call the function to combine CSV files
    combined_data = combine_csv_files(folder_path)

    # Calculate means and medians
    combined_data = calculate_summary(combined_data)

    # Display the combined DataFrame
    make_figures(combined_data, rho, 8, test)