import time

import pandas as pd

from src.classes import *
from src.simulate_lorenz import *
from matplotlib.backends.backend_pdf import PdfPages
import os


def get_parameters_from_name(file):
    #f"length = {length}, noise = {noise}, n_repl = {n_repl}.csv")
    # Length
    length = file[9:12]
    length = length.replace(",", "")
    length = int(length)

    # noise
    noise_i = file.find('noise = ') + 8
    noise = file[noise_i:noise_i + 3]
    noise = float(noise)

    # var
    var_i = file.find('var = ') + 6
    var = file[var_i:var_i + 2]
    var = var.replace(".", "")
    var = float(var)

    return length, noise, var

def make_df(rho, test):
    df = pd.DataFrame()
    path = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
            f"RMSE vs n_repl/rho = {rho}/{test}")

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        file_df = pd.read_csv(file_path)
        file_df = file_df.dropna(how='any')

        len, noise, var = get_parameters_from_name(filename)

        for n_repl in [0, 1, 2, 4, 8, 16, 32]:
            sub_df = file_df[file_df['n_repl'] == n_repl]

            # get 10th and 90th percentiles and mean RMSE
            percentiles = sub_df['RMSE'].describe(percentiles=[0.1, 0.9]).loc[['mean', '10%', '90%']]

            row = pd.DataFrame({'length': [len], 'noise': [noise], 'n_repl': [n_repl], 'variance': [var],
                                '10th': [percentiles['10%']], '90th': [percentiles['90%']], 'mean_RMSE': [percentiles['mean']]})
            df = pd.concat([df, row])

    return df

def plot_RMSE_vs_n_repl(df, rho, test):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for length in df['length'].unique():
        for noise in df['noise'].unique():
            i = 0
            for var in df['variance'].unique():
                color = colors(i)
                marker = markers[i]

                sub_df = df[(df['length'] == length) & (df['noise'] == noise) & (df['variance'] == var)]

                plt.plot(sub_df['n_repl'], sub_df['mean_RMSE'], color=color)
                plt.scatter(sub_df['n_repl'], sub_df['mean_RMSE'], color=color, marker=marker, label=f'{var}')

                plt.fill_between(sub_df['n_repl'], sub_df['10th'], sub_df['90th'], color=color, alpha = .2)
                i += 1

            plt.legend(title='variance')
            plt.title(f"length = {length} - noise = {noise}")
            plt.savefig(f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/RMSE vs n_repl/Figures/"
                        f"{rho}, {test}, {length}, {noise}.png")
            plt.show()

            time.sleep(2)



if __name__ == '__main__':

    df_28  = make_df(28, 'begin_conditions')
    plot_RMSE_vs_n_repl(df_28, 28, 'begin_conditions')









