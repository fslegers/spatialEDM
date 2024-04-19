import os
from itertools import product

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def calculate_performance_measures(obs, pred):
    RMSE = np.sqrt(np.mean((obs - pred)**2))
    MAE = np.mean(np.abs(obs - pred))
    corr = pearsonr(obs, pred)[0]
    return RMSE, MAE, corr


def plot_predictions(df):

    for hor in range(1, 6):
        for init_var in [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            df_sub = df[(df['hor'] == hor) & (df['init_var'] == init_var)]
            plt.scatter(df_sub['obs'], df_sub['pred_smap'])
            plt.title("Horizon: " + str(hor) + ", variance: " + str(init_var))
            plt.xlabel('observed')
            plt.ylabel('predicted')
            plt.axis('equal')
            plt.plot()

    return 0


def get_parameters_from_name(file):
    # Length
    len = file[18:21]
    len = len.replace(",", "")
    len = int(len)

    # noise
    noise_i = file.find('noise = ') + 8
    noise = file[noise_i:noise_i + 3]
    noise = float(noise)

    return len, noise


def calculate_RMSEs(rho):
    # Load all CSV files from the folder into a pandas DataFrame
    folder_path = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}")

    df = pd.DataFrame({'training_length': [], 'noise': [], 'hor': [], 'iter': [], 'RMSE': [], 'MAE': [], 'corr': []})
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv') and filename.startswith('training'):
            file_path = os.path.join(folder_path, filename)
            result = pd.read_csv(file_path)
            result = result.dropna(how='any')

            len, noise = get_parameters_from_name(filename)

            for hor, iter in product(range(1, 11), range(200)):
                observed = result[(result['hor'] == hor) & (result['iter'] == iter)]['obs']
                predicted = result[(result['hor'] == hor) & (result['iter'] == iter)]['pred']
                RMSE, MAE, corr = calculate_performance_measures(observed, predicted)
                new_row = pd.DataFrame({'training_length': [len], 'noise': [noise], 'hor': [hor], 'iter': [iter],
                                        'RMSE': [RMSE], 'MAE': [MAE], 'corr': [corr]})
                df = pd.concat([df, new_row], ignore_index=True)

    # # plot predictions
    # plot_predictions(df)

    save_path = f'C:/Users/fleur/Documents/Resultaten/summarized'
    save_path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/summarized.csv"
    df.to_csv(save_path)


def summarize_results(rho):
    folder_path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/summarized.csv"

    df = pd.read_csv(folder_path, index_col=False)
    df = df.iloc[:, 1:]
    df = df.drop(['iter'], axis=1)

    min = df.groupby(['training_length', 'noise', 'hor'], as_index=False).quantile(0.1) # 10%quantile
    min = min.rename(columns={'RMSE': 'min_RMSE', 'MAE': 'min_MAE', 'corr': 'min_corr'})

    max = df.groupby(['training_length', 'noise', 'hor'], as_index=False).quantile(0.9) # 90% quantile
    max = max.rename(columns={'RMSE': 'max_RMSE', 'MAE': 'max_MAE', 'corr': 'max_corr'})

    mean = df.groupby(['training_length', 'noise', 'hor'], as_index=False).mean()
    mean = mean.rename(columns={'RMSE': 'mean_RMSE', 'MAE': 'mean_MAE', 'corr': 'mean_corr'})

    df = min.merge(max, how='inner', on=['training_length', 'noise', 'hor'])
    df = df.merge(mean, how='inner', on=['training_length', 'noise', 'hor'])

    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}/mean and percentiles.csv"
    df.to_csv(path)


if __name__ == "__main__":

    rho = 28
    calculate_RMSEs(rho)
    summarize_results(rho)











