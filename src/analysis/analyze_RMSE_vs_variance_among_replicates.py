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

    # n_repl
    n_repl_i = file.find('n_repl = ') + 9
    n_repl = file[n_repl_i:n_repl_i + 2]
    n_repl = n_repl.replace(".", "")
    n_repl = int(n_repl)

    return length, noise, n_repl

def make_df(rho, test):
    df = pd.DataFrame()
    # path = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
    #         f"RMSE vs variance among replicates/rho = {rho}/{test}")
    path = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
            f"RMSE vs variance among replicates/test_2/rho = {rho}")

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        file_df = pd.read_csv(file_path)
        file_df = file_df.dropna(how='any')

        len, noise, n_repl = get_parameters_from_name(filename)

        for variance in np.arange(0, 50, 2.5):
            sub_df = file_df[file_df['variance'] == variance]

            # get 10th and 90th percentiles and mean RMSE
            percentiles = sub_df['RMSE'].describe(percentiles=[0.1, 0.9]).loc[['mean', '10%', '90%']]

            row = pd.DataFrame({'length': [len], 'noise': [noise], 'n_repl': [n_repl], 'variance': [variance],
                                '10th': [percentiles['10%']], '90th': [percentiles['90%']], 'mean_RMSE': [percentiles['mean']]})
            df = pd.concat([df, row])

    return df

def plot_RMSE_vs_variance(df, test, rho):
    colors = plt.get_cmap('tab10')
    markers = ['s', 'o', 'v']

    for length in df['length'].unique():
        for n_repl in df['n_repl'].unique():
            i = 0
            for noise in df['noise'].unique():
                color = colors(i)
                marker = markers[i]

                sub_df = df[(df['length'] == length) & (df['noise'] == noise) & (df['n_repl'] == n_repl)]

                plt.plot(sub_df['variance'], sub_df['mean_RMSE'], color=color)
                plt.scatter(sub_df['variance'], sub_df['mean_RMSE'], color=color, marker=marker, label=f'{noise}')

                plt.fill_between(sub_df['variance'], sub_df['10th'], sub_df['90th'], color=color, alpha = .2)
                i += 1

            plt.legend(title='noise')
            plt.title(f"test = {test} - length = {length} - n_repl = {n_repl}")
            plt.savefig(f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/RMSE vs variance among replicates/Figures/"
                        f"{rho}, {test}, {length}, {n_repl}")
            plt.show()

            time.sleep(2)

# def plot_6(df_28, df_20):
#
#     colors = plt.get_cmap('tab10')
#     markers = ['s', 'o', 'v']
#     plt.rc('font', size=16)
#
#     fig, axs = plt.subplots(2, 3, figsize=(16, 8))
#     plt.subplots_adjust(hspace=1.3, wspace=0.0)
#
#     df = df_28
#     # First plot
#     # Length = 25, RMSE vs horizon
#     df_sub = df[df['training_length'] == 25]
#
#     i = 0
#     for noise in [0.0, 2.0, 4.0]:
#         color = colors(i)
#         marker = markers[i]
#         df_noise = df_sub[df_sub['noise'] == noise]
#
#         if noise == 0.0:
#             ax1 = axs[0,0].twinx()
#             ax1.fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
#             ax1.plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#             ax1.plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#             ax1.plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#             ax1.scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
#             #ax1.set_ylabel('RMSE for noise=0.0')
#             ax1.set_ylim(0, max(df_sub[df_sub['noise'] == 0.0]['mean_RMSE']) * 1.1)  # Adjust the multiplier as needed
#             ax1.tick_params(axis='y', colors=color)
#         else:
#             axs[0,0].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
#             axs[0,0].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#             axs[0,0].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#             axs[0, 0].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#             axs[0, 0].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
#         i += 1
#
#     #axs[0,0].set_xlabel('horizon)
#     axs[0,0].set_ylabel('RMSE')
#
#     # Second plot
#     # Length = 75, RMSE vs horizon
#     df_sub = df[df['training_length'] == 75]
#
#     i = 0
#     for noise in [0.0, 2.0, 4.0]:
#         color = colors(i)
#         marker = markers[i]
#         df_noise = df_sub[df_sub['noise'] == noise]
#         axs[0, 1].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
#         axs[0, 1].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[0, 1].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[0, 1].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#         axs[0, 1].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
#         i += 1
#
#     #axs[0, 1].set_xlabel('horizon)
#     #axs[0, 1].set_ylabel('RMSE')
#
#     # third plot
#     # RMSE vs length
#     i = 0
#     for noise in [0.0, 2.0, 4.0]:
#         color = colors(i)
#         marker = markers[i]
#         df_noise = df[(df['noise'] == noise) & (df['hor'] == 1)]
#         axs[0, 2].fill_between(df_noise['training_length'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
#         axs[0, 2].plot(df_noise['training_length'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[0, 2].plot(df_noise['training_length'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[0, 2].plot(df_noise['training_length'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#         axs[0, 2].scatter(df_noise['training_length'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
#         i += 1
#
#     # Set labels and title
#     #axs[0, 2].set_xlabel('Training length')
#     #axs[0, 2].set_ylabel('RMSE')
#
#     df = df_20
#     # 4th plot
#     # Length = 25, RMSE vs horizon
#     df_sub = df[df['training_length'] == 25]
#
#     i = 0
#     for noise in [0.0, 2.0, 4.0]:
#         color = colors(i)
#         marker = markers[i]
#         df_noise = df_sub[df_sub['noise'] == noise]
#         axs[1, 0].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
#         axs[1, 0].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[1, 0].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[1, 0].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#         axs[1, 0].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
#         i += 1
#
#     axs[1, 0].set_xlabel('horizon')
#     axs[1, 0].set_ylabel('RMSE')
#
#     # 5th plot
#     # Length = 75, RMSE vs horizon
#     df_sub = df[df['training_length'] == 75]
#
#     i = 0
#     for noise in [0.0, 2.0, 4.0]:
#         color = colors(i)
#         marker = markers[i]
#         df_noise = df_sub[df_sub['noise'] == noise]
#         axs[1, 1].fill_between(df_noise['hor'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1, color=color)
#         axs[1, 1].plot(df_noise['hor'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[1, 1].plot(df_noise['hor'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[1, 1].plot(df_noise['hor'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#         axs[1, 1].scatter(df_noise['hor'], df_noise['mean_RMSE'], color=color, marker=marker, s=60)
#         i += 1
#
#     axs[1, 1].set_xlabel('horizon')
#     #axs[1, 1].set_ylabel('RMSE')
#
#     # 6th plot
#     # RMSE vs length
#     i = 0
#     for noise in [0.0, 2.0, 4.0]:
#         color = colors(i)
#         marker = markers[i]
#         df_noise = df[(df['noise'] == noise) & (df['hor'] == 1)]
#         axs[1, 2].fill_between(df_noise['training_length'], df_noise['min_RMSE'], df_noise['max_RMSE'], alpha=.1,color=color)
#         axs[1, 2].plot(df_noise['training_length'], df_noise['min_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[1, 2].plot(df_noise['training_length'], df_noise['max_RMSE'], color=color, linewidth=1, alpha=.8)
#         axs[1, 2].plot(df_noise['training_length'], df_noise['mean_RMSE'], label=f'{noise}', color=color, linewidth=2.0)
#         axs[1, 2].scatter(df_noise['training_length'], df_noise['mean_RMSE'], color=color, marker=marker, s=50)
#         i += 1
#
#     # Set labels and title
#     axs[1, 2].set_xlabel('training length')
#     #axs[1, 2].set_ylabel('RMSE')
#
#
#     # Add (a) - (f)
#     axs[0, 0].text(0.5, -0.25, '(a)', transform=axs[0, 0].transAxes)
#     axs[0, 1].text(0.5, -0.25, '(b)', transform=axs[0, 1].transAxes)
#     axs[0, 2].text(0.5, -0.25, '(c)', transform=axs[0, 2].transAxes)
#     axs[1, 0].text(0.5, -0.35, '(d)', transform=axs[1, 0].transAxes)
#     axs[1, 1].text(0.5, -0.35, '(e)', transform=axs[1, 1].transAxes)
#     axs[1, 2].text(0.5, -0.35, '(f)', transform=axs[1, 2].transAxes)
#
#     axs[0, 0].text(-0.3, 0.37, r'$\rho = 28$', transform=axs[0, 0].transAxes, fontsize=16, rotation=90)
#     axs[1, 0].text(-0.3, 0.37, r'$\rho = 20$', transform=axs[1, 0].transAxes, fontsize=16, rotation=90)
#
#     # axs[0, 0].text(1.0, 1.3, r'Training length', transform=axs[0, 0].transAxes, fontsize=18)
#     # axs[0, 0].text(0.5, 1.1, r'$25$', transform=axs[0, 0].transAxes, fontsize=16)
#     # axs[0, 1].text(0.5, 1.1, r'$75$', transform=axs[0, 1].transAxes, fontsize=16)
#
#     fig.tight_layout(pad=2.1, h_pad=3, w_pad=0)
#
#     path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/6 plots.png"
#     plt.savefig(path, dpi = 500)
#     plt.show()

# def make_pdf(rho):
#     path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/rho = {rho}"
#     folders = [path+'/RMSE, horizon', path+'/RMSE, length', path+'/RMSE, noise']
#
#     # Calculate number of pages needed for each folder
#     num_pages_per_folder = [len(os.listdir(folder)) / 12 + 1 for folder in folders]
#     num_pages_per_folder[0] -= 1
#
#     # Initialize matplotlib figure
#     figsize = (8.5, 11)  # A4 paper size
#     pdf_pages = PdfPages(path+"/Figures for Appendix.pdf")
#
#     for folder, num_pages in zip(folders, num_pages_per_folder):
#         for page_num in range(int(num_pages)):
#             # Initialize matplotlib figure for each page
#             fig, axs = plt.subplots(4, 3, figsize=(17, 22))
#             fig.subplots_adjust(wspace=0.0, hspace=0.0)
#
#             # Calculate start and end index for files to display on this page
#             start_index = page_num * 12
#             end_index = min((page_num + 1) * 12, len(os.listdir(folder)))
#
#             # Display each file on this page
#             for i in range(4):
#                 for j in range(3):
#                     img_index = start_index + i * 3 + j
#                     if img_index < end_index:
#                         img_path = os.path.join(folder, os.listdir(folder)[img_index])
#                         img = plt.imread(img_path)
#                         axs[i, j].imshow(img)
#                     axs[i, j].axis('off')
#
#             # Save the figure to the PDF
#             pdf_pages.savefig(fig, dpi=300)
#             plt.close(fig)
#
#     # Close the PDF
#     pdf_pages.close()



if __name__ == '__main__':

    df_28_IC  = make_df(28, 'begin_conditions')
    # df_28_rho = make_df(28, 'rho')
    # df_20_IC  = make_df(20, 'begin_conditions')
    # df_20_rho = make_df(20, 'rho')
    plot_RMSE_vs_variance(df_28_IC, 'IC', 28)
    # plot_RMSE_vs_variance(df_28_rho, 'Rho', 28)
    # plot_RMSE_vs_variance(df_20_IC)
    # plot_RMSE_vs_variance(df_20_rho)









