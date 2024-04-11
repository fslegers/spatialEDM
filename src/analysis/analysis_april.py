import os
import pandas as pd
import numpy as np
import random
from scipy.stats import pearsonr
from matplotlib import pyplot as plt

def calculate_performance_measures(df):
    obs = df['obs'].values
    pred = df['pred'].values

    RMSE = np.sqrt(np.mean((obs - pred)**2))
    MAE = np.mean(np.abs(obs - pred))
    corr = pearsonr(obs, pred)[0]

    return RMSE, MAE, corr

def get_parameter_values(file):

    # Length
    len = file[18:21]
    len = len.replace(",", "")
    len = int(len)

    # n_replicates
    repl_i = file.find('n_repl = ')
    if repl_i < 0:
        repl = 0
    else:
        repl_i += 9
        repl = file[repl_i:repl_i+2]
        repl = repl.replace(",", "")
        repl = int(repl)

    # noise
    noise_i = file.find('noise = ')+8
    noise = file[noise_i:noise_i+3]
    noise = float(noise)

    # variance
    var_i = file.find('var = ')+6
    var = file[var_i:var_i + 4]
    var = var.replace(",", "")
    var = float(var)

    # horizon
    hor_i = file.find("hor = ")+6
    hor = file[hor_i]
    hor = int(hor)

    return len, repl, noise, var, hor

def summarize_results(test, rho):

    results = pd.DataFrame({'length':[], 'n_repl': [], 'noise':[], 'var':[], 'hor':[], 'iter':[],
                            'RMSE': [], 'corr': [], 'MAE': []})

    folder = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/{test}/rho = {rho}"

    # Load all CSV files from the folder into a pandas DataFrame
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        df = pd.read_csv(path)
        df = df.dropna(how='any')

        RMSE, corr, MAE = calculate_performance_measures(df)
        len, repl, noise, var, hor = get_parameter_values(file)

        new_row = pd.DataFrame({'length': [len], 'n_repl': [repl], 'noise': [noise], 'var': [var], 'hor': [hor], 'iter': [iter],
                                'RMSE': [RMSE], 'corr': [corr], 'MAE': [MAE]})

        results = pd.concat([results, new_row], ignore_index=True)
        
    return(results)

def plot_results(df):
    plot_RMSE_vs_horizon(df)
    plot_RMSE_vs_noise(df)
    plot_RMSE_vs_variance(df)

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


if __name__ == "__main__":

    random.seed(123)

    test = "begin conditions"
    rho = 28

    df = summarize_results(test, rho)
    plot_results(df)



