import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def calculate_performance_measures(obs, pred):
    RMSE = np.sqrt(np.mean((obs - pred)**2))
    MAE = np.mean(np.abs(obs - pred))
    corr = pearsonr(obs, pred)[0]
    return RMSE, MAE, corr


def calculate_confidence_intervals(observed, predicted):
    return 0


def plot_predictions(df):

    for hor in range(1, 6):
        for init_var in [0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
            df_sub = df[(df['hor'] == hor) & (df['init_var'] == init_var)]
            plt.scatter(df_sub['obs'], df_sub['pred_smap'])
            plt.title("Horizon: " + str(hor) + ", variance: " + str(init_var))
            plt.xlabel('observed')
            plt.ylabel('predicted')
            plt.axis('equal')
            plt.pause(2)

    return 0


def get_parameters_from_name(file):
    # Length
    len = file[18:21]
    len = len.replace(",", "")
    len = int(len)

    # n_replicates
    repl_i = file.find('n_repl = ') + 9
    repl = file[repl_i:repl_i + 2]
    repl = repl.replace(",", "")
    repl = int(repl)

    # noise
    noise_i = file.find('noise = ') + 8
    noise = file[noise_i:noise_i + 3]
    noise = float(noise)

    # variance
    var_i = file.find('var = ') + 6
    var = file[var_i:var_i + 4]
    var = var.replace(",", "")
    var = float(var)

    # horizon
    hor_i = file.find("hor = ") + 6
    hor = file[hor_i : hor_i + 1]
    hor = hor.replace(",", "")
    hor = int(hor)

    # iteration
    iter_i = file.find("iter = ") + 7
    iter = file[iter_i:]
    iter = iter.replace(".csv", "")
    iter = int(iter)

    return len, repl, noise, var, hor, iter



if __name__ == "__main__":

    test = "begin_conditions"
    rho = 20

    # Load all CSV files from the folder into a pandas DataFrame
    folder_path = f'C:/Users/fleur/Documents/Resultaten/{test}/rho = {rho}'
    df = pd.DataFrame({'training_length': [], 'n_replicates': [], 'noise': [], 'var': [], 'hor': [], 'iter': [],
                       'RMSE': [], 'MAE': [], 'corr': []})

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        result = pd.read_csv(file_path)
        result = result.dropna(how='any')

        len, repl, noise, var, hor, iter = get_parameters_from_name(filename)
        RMSE, MAE, corr = calculate_performance_measures(result['obs'], result['pred'])

        new_row = pd.DataFrame({'training_length': [len], 'n_replicates': [repl], 'noise': [noise], 'var': [var],
                                'hor': [int(hor)], 'iter': [int(iter)], 'RMSE': [RMSE], 'MAE': [MAE], 'corr': [corr]})

        df = pd.concat([df, new_row], ignore_index=True)

    save_path = f'C:/Users/fleur/Documents/Resultaten/summarized'
    save_name = save_path + f'/{test}, rho = {rho}.csv'
    df.to_csv(save_name)

    # Part 2!
    folder_path = f'C:/Users/fleur/Documents/Resultaten/summarized/{test}, rho = {rho}.csv'
    df = pd.read_csv(folder_path, index_col=False)
    df = df.iloc[:, 1:]
    df = df.drop(['iter'], axis=1)

    min = df.groupby(['training_length', 'n_replicates', 'noise', 'var', 'hor'], as_index=False).min()
    min = min.rename(columns={'RMSE': 'min_RMSE', 'MAE': 'min_MAE', 'corr': 'min_corr'})

    max = df.groupby(['training_length', 'n_replicates', 'noise', 'var', 'hor'], as_index=False).max()
    max = max.rename(columns={'RMSE': 'max_RMSE', 'MAE': 'max_MAE', 'corr': 'max_corr'})

    mean = df.groupby(['training_length', 'n_replicates', 'noise', 'var', 'hor'], as_index=False).mean()
    mean = mean.rename(columns={'RMSE': 'mean_RMSE', 'MAE': 'mean_MAE', 'corr': 'mean_corr'})

    df = min.merge(max, how='inner', on=['training_length', 'n_replicates', 'noise', 'var', 'hor'])
    df = df.merge(mean, how='inner', on=['training_length', 'n_replicates', 'noise', 'var', 'hor'])

    save_path = 'C:/Users/fleur/Documents/Resultaten/summarized'
    save_name = save_path + f'/{test}, rho = {rho}, min max and mean measures.csv'
    df.to_csv(save_name)








