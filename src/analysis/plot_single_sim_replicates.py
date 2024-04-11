import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


def plot_RMSE_vs_horizon(df, rho, test):
    colors = plt.get_cmap('tab10')

    for len, var, repl in product([25, 50, 75, 100], [1.0, 4.0, 7.0, 10.0, 13.0], [1, 2, 4, 8]):

        df_sub = df[(df['training_length'] == len) & (df['var'] == var) & (df['n_replicates'] == repl)]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0]:
            color = colors(i)
            df_noise = df_sub[df_sub['noise'] == noise]
            plt.plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color)
            plt.scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color)
            plt.plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Horizon')
        plt.ylabel('RMSE')
        plt.title(f'len = {len}, var = {var}, n_repl = {repl}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, RMSE, horizon-plot (len={len},var={var}, repl={repl},noise={noise}).png"

        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


def plot_corr_vs_horizon(df, rho, test):
    colors = plt.get_cmap('tab10')

    for len, var, repl in product([25, 50, 75, 100], [1.0, 4.0, 7.0, 10.0, 13.0], [1, 2, 4, 8]):

        df_sub = df[(df['training_length'] == len) & (df['var'] == var) & (df['n_replicates'] == repl)]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0]:
            color = colors(i)
            df_noise = df_sub[df_sub['noise'] == noise]
            plt.plot(df_noise['hor'], df_noise['mean_corr'], label=f'{noise}', color=color)
            plt.scatter(df_noise['hor'], df_noise['mean_corr'], color=color)
            plt.plot(df_noise['hor'], df_noise['min_corr'], color=color, linestyle="--")
            plt.plot(df_noise['hor'], df_noise['max_corr'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Horizon')
        plt.ylabel('corr')
        plt.title(f'len = {len}, var = {var}, n_repl = {repl}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, corr, horizon-plot (len={len},var={var}, repl={repl},noise={noise}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


def plot_RMSE_vs_noise(df, rho, test):
    colors = plt.get_cmap('tab10')

    for len, var, repl in product([25, 50, 75, 100], [1.0, 4.0, 7.0, 10.0, 13.0], [1, 2, 4, 8]):

        df_sub = df[(df['training_length'] == len) & (df['var'] == var) & (df['n_replicates'] == repl)]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for hor in np.arange(1, 11):
            color = colors(i)
            df_hor = df_sub[df_sub['hor'] == hor]
            plt.plot(df_hor['noise'], df_hor['mean_RMSE'], label=f'{hor}', color=color)
            plt.scatter(df_hor['noise'], df_hor['mean_RMSE'], color=color)
            plt.plot(df_hor['noise'], df_hor['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_hor['noise'], df_hor['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Noise level')
        plt.ylabel('RMSE')
        plt.title(f'len = {len}, var = {var}, n_repl = {repl}')
        plt.legend(title='Horizon')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, RMSE, noise-plot (len={len},var={var}, repl={repl},hor={hor}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


def plot_corr_vs_noise(df, rho, test):
    colors = plt.get_cmap('tab10')

    for len, var, repl in product([25, 50, 75, 100], [1.0, 4.0, 7.0, 10.0, 13.0], [1, 2, 4, 8]):

        df_sub = df[(df['training_length'] == len) & (df['var'] == var) & (df['n_replicates'] == repl)]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for hor in np.arange(1, 11):
            color = colors(i)
            df_hor = df_sub[df_sub['hor'] == hor]
            plt.plot(df_hor['noise'], df_hor['mean_corr'], label=f'{hor}', color=color)
            plt.scatter(df_hor['noise'], df_hor['mean_corr'], color=color)
            plt.plot(df_hor['noise'], df_hor['min_corr'], color=color, linestyle="--")
            plt.plot(df_hor['noise'], df_hor['max_corr'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Noise level')
        plt.ylabel('corr')
        plt.title(f'len = {len}, var = {var}, n_repl = {repl}')
        plt.legend(title='Horizon')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, corr, noise-plot (len={len},var={var}, repl={repl},hor={hor}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


def plot_RMSE_vs_len(df, rho, test):
    colors = plt.get_cmap('tab10')

    for var, repl, hor in product([1.0, 4.0, 7.0, 10.0, 13.0], [1, 2, 4, 8], np.arange(1,10)):
        df_sub = df[(df['hor'] == hor) & (df['var'] == var) & (df['n_replicates'] == repl)]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0]:
            color = colors(i)
            df_noise = df_sub[df_sub['noise'] == noise]
            plt.plot(df_noise['training_length'], df_noise['mean_RMSE'], label=f'{noise}', color=color)
            plt.scatter(df_noise['training_length'], df_noise['mean_RMSE'], color=color)
            plt.plot(df_noise['training_length'], df_noise['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_noise['training_length'], df_noise['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Training length')
        plt.ylabel('RMSE')
        plt.title(f'hor = {hor}, var = {var}, n_repl = {repl}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, RMSE, length-plot (hor={hor},var={var}, repl={repl}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


def plot_corr_vs_len(df, rho, test):
    colors = plt.get_cmap('tab10')

    for var, repl, hor in product([1.0, 4.0, 7.0, 10.0, 13.0], [1, 2, 4, 8], np.arange(1,10)):
        df_sub = df[(df['hor'] == hor) & (df['var'] == var) & (df['n_replicates'] == repl)]
        plt.figure()

        # Iterate over groups and plot corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png
        i = 0
        for noise in [0.0, 1.0, 2.0, 3.0, 4.0]:
            color = colors(i)
            df_noise = df_sub[df_sub['noise'] == noise]
            plt.plot(df_noise['training_length'], df_noise['mean_corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png'], label=f'{noise}', color=color)
            plt.scatter(df_noise['training_length'], df_noise['mean_corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png'], color=color)
            plt.plot(df_noise['training_length'], df_noise['min_corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png'], color=color, linestyle="--")
            plt.plot(df_noise['training_length'], df_noise['max_corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('Training length')
        plt.ylabel('corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png')
        plt.title(f'hor = {hor}, var = {var}, n_repl = {repl}')
        plt.legend(title='Noise')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, corr, horizon-plot (len=25,var=1.0, repl=1,noise=4.0).png, length-plot (hor={hor},var={var}, repl={repl}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()


def plot_RMSE_vs_variance(df, rho, test):
    colors = plt.get_cmap('tab10')

    for len, noise, hor in product([25, 50, 75, 100], [0.0, 1.0, 2.0, 3.0, 4.0], np.arange(1,10)):
        df_sub = df[(df['hor'] == hor) & (df['training_length'] == len) & (df['noise'] == noise)]
        plt.figure()

        # Iterate over groups and plot RMSE
        i = 0
        for var in [1.0, 4.0, 7.0, 10.0, 13.0]:
            color = colors(i)
            df_var = df_sub[df_sub['var'] == var]
            plt.plot(df_var['n_replicates'], df_var['mean_RMSE'], label=f'{var}', color=color)
            plt.scatter(df_var['n_replicates'], df_var['mean_RMSE'], color=color)
            plt.plot(df_var['n_replicates'], df_var['min_RMSE'], color=color, linestyle="--")
            plt.plot(df_var['n_replicates'], df_var['max_RMSE'], color=color, linestyle="--")
            i += 1

        # Set labels and title
        plt.xlabel('# replicates')
        plt.ylabel('RMSE')
        plt.title(f'hor = {hor}, len = {len}, noise = {noise}')
        plt.legend(title='Variance')
        plt.tight_layout()

        # Show or save the plot
        file_path = f"C:/users/fleur/Documents/Resultaten/summarized/figures/rho={rho}, test={test}, RMSE, variance-plot (hor={hor},len={len}, noise={noise}).png"
        plt.savefig(file_path)
        time.sleep(2)
        plt.show()



if __name__ == '__main__':

    test = "begin_conditions"
    rho = 20

    # Load CSV file into a pandas DataFrame
    path = 'C:/Users/fleur/Documents/Resultaten/summarized'
    path = path + f'/{test}, rho = {rho}, min max and mean measures.csv'

    df = pd.read_csv(path)

    plot_RMSE_vs_horizon(df, rho, test)
    #plot_corr_vs_horizon(df, rho, test)

    plot_RMSE_vs_noise(df, rho, test)
    #plot_corr_vs_noise(df, rho, test)

    plot_RMSE_vs_len(df, rho, test)
    #plot_corr_vs_len(df, rho, test)

    plot_RMSE_vs_variance(df, rho, test)
    #plot_corr_vs_variance(df, rho, test)







