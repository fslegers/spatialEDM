import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_RMSE_vs_horizon(df):
    # Get colors
    colors = plt.get_cmap('tab10')

    # Group by unique combinations of obs_noise, hor, and init_var
    grouped = df.groupby('obs_noise')

    # Iterate over groups and create plots
    for obs_noise, group_df in grouped:
        # Group by init_var for each obs_noise
        obs_noise_grouped = group_df.groupby('init_var')

        # Create a figure for the plot
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for init_var, init_var_group in obs_noise_grouped:
            color = colors(i)
            plt.plot(init_var_group['hor'], init_var_group['RMSE'], label=f'{init_var}', color=color)
            plt.scatter(init_var_group['hor'], init_var_group['RMSE'], color=color)
            plt.plot(init_var_group['hor'], init_var_group['upper'], color=color, linestyle="--")
            plt.plot(init_var_group['hor'], init_var_group['lower'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Horizon')
        plt.ylabel('RMSE')
        plt.title(f'Observational Noise: {obs_noise}')
        plt.legend(title='Variance among \n replicates')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"{folder_path}/figures/RMSE, horizon-plot (noise={obs_noise}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()

def plot_RMSE_vs_noise(df):
    # Group by unique combinations of obs_noise, hor, and init_var
    grouped = df.groupby('init_var')

    # Iterate over groups and create plots
    for init_var, group_df in grouped:
        # Group by hor for each init_var
        init_var_grouped = group_df.groupby('hor')

        # Create a figure for the plot
        plt.figure()

        # Iterate over groups and plot RMSE
        for hor, hor_group in init_var_grouped:
            plt.plot(hor_group['obs_noise'], hor_group['RMSE'], label=f'{hor}')
            plt.scatter(hor_group['obs_noise'], hor_group['RMSE'])

        # Set labels and title
        plt.xlabel('Observational noise')
        plt.ylabel('RMSE')
        plt.title(f'Variance among replicates: {init_var}')
        plt.legend(title='Horizon')

        # Show or save the plot
        file_path = f"{folder_path}/figures/RMSE, noise-plot (variance={init_var}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()

def plot_RMSE_vs_variance(df):
    # Group by unique combinations of obs_noise, init_var, and hor
    grouped = df.groupby('obs_noise')

    # Iterate over groups and create plots
    for obs_noise, group_df in grouped:
        # Group by hor for each obs_noise
        obs_noise_grouped = group_df.groupby('hor')

        # Create a figure for the plot
        plt.figure()

        # Iterate over groups and plot RMSE
        for hor, hor_group in obs_noise_grouped:
            plt.plot(hor_group['init_var'], hor_group['RMSE'], label=f'{hor}')
            plt.scatter(hor_group['init_var'], hor_group['RMSE'])

        # Set labels and title
        plt.xlabel('init_var')
        plt.ylabel('RMSE')
        plt.title(f'Observational Noise: {obs_noise}')
        plt.legend(title='Horizon')

        # Show or save the plot
        file_path = f"{folder_path}/figures/RMSE, variance-plot (noise={obs_noise}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


if __name__ == '__main__':
    # Load CSV file into a pandas DataFrame
    folder_path  = 'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten'
    folder_path = folder_path + '/04-03-24'
    file_path = folder_path + '/summarized/results.csv'
    df = pd.read_csv(file_path)

    # Delete all horizon = 5 because there is no data for it
    #df = df[df.hor < 5]

    plot_RMSE_vs_horizon(df)
    plot_RMSE_vs_noise(df)
    plot_RMSE_vs_variance(df)







