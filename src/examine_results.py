import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os

path_name = "rho=28 EDM LB_CV 250sim long_test_set"
os.chdir("C:/Users/5605407/Documents/PhD/PythonProjects/timeseriesanalysis/results/output/"+path_name)
folders = os.listdir()

intervals = [1, 5, 10]
lengths = [15, 20, 25, 30, 40, 50, 75, 100, 150, 300]

stds = [0, 0.05, 0.1, 0.5, 1, 2.5, 5, 10]
stds_strings = ['0', '0.05', '0.5', '0.1', '1', '2.5', '5', '10']
sd_mean = np.zeros((len(lengths), 8))
sd_std = np.zeros((len(lengths), 8))

# Each folder has a different level of noise (sd)
for folder in folders:

    # Which sd is this folder?
    strings = folder.split(" = ")
    sd = strings[1]

    # List all files in folder
    files = os.listdir("./" + folder)

    # Create matrices to save the results
    mean_test = np.zeros((len(intervals), len(lengths)))
    std_test = np.zeros((len(intervals), len(lengths)))
    mean_training = np.zeros((len(intervals), len(lengths)))
    std_training = np.zeros((len(intervals), len(lengths)))

    # For each combination of interval + length, collect all results
    for i in range(len(intervals)):
        for j in range(len(lengths)):
            values_test = []
            values_training = []

            for file in files:
                path = "./" + folder + "/" + file
                df = pd.read_csv(path, index_col=0)

                strings = file.split(", ")
                train_or_test = strings[1]
                train_or_test = train_or_test.split(".")[0]

                if train_or_test == "test":
                    values_test.append(df.iloc[i][j])
                elif train_or_test == "training":
                    values_training.append(df.iloc[i][j])

            mean_test[i,j] = np.nanmean(values_test)
            mean_training[i,j] = np.nanmean(values_training)
            std_test[i,j] = np.nanstd(values_test)
            std_training[i,j] = np.nanstd(values_training)

            if i == 1:
                # Save result in matrix with sds
                sd_mean[j, stds_strings.index(sd)] = mean_test[i,j]
                sd_std[j, stds_strings.index(sd)] = std_test[i,j]


    # Start plotting
    cmap = matplotlib.colormaps['Blues']

    # Plot test results
    for row in range(len(intervals) - 1):
        col = cmap((len(intervals) - row)/len(intervals))
        plt.plot(lengths, mean_test[row,:], color=col, label=str(intervals[row]))
        plt.scatter(lengths, mean_test[row,:], color=col)

        plt.fill_between(lengths, (mean_test[row,:] - std_test[row,:]),
                         (mean_test[row,:] + std_test[row,:]), color=col, alpha=.1)

        plt.ylim((-0.3, 1.1))
        plt.xlabel("Time series length")
        plt.ylabel("Correlation coefficient")
        plt.ylabel("Correlation coefficient")
        plt.title("Performance on test data \nStandard deviation: " + sd)
        plt.legend(title = "Sampling interval", loc='center left', bbox_to_anchor=(1, 0.5))

    path = "C:/Users/5605407/Documents/PhD/PythonProjects/timeseriesanalysis/results/figures/" + path_name + "/test"

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/figure_sd_" + str(sd).replace(".", ","), bbox_inches='tight')

    plt.show()

    # Plot training results
    for row in range(len(intervals) - 1):
        col = cmap((len(intervals) - row)/len(intervals))
        plt.plot(lengths, mean_training[row,:], color=col, label=str(intervals[row]))
        plt.scatter(lengths, mean_training[row,:], color=col)

        plt.fill_between(lengths, (mean_training[row,:] - std_training[row,:]),
                         (mean_training[row,:] + std_training[row,:]), color=col, alpha=.1)

        plt.ylim((-0.3, 1.1))
        plt.xlabel("Time series length")
        plt.ylabel("Correlation coefficient")
        plt.title("Performance on training data \n Standard deviation: " + sd)
        plt.legend(title = "Sampling interval", loc='center left', bbox_to_anchor=(1, 0.5))

    path = "C:/Users/5605407/Documents/PhD/PythonProjects/timeseriesanalysis/results/figures/"+path_name+"/training"

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + "/figure_sd_" + str(sd).replace(".", ","), bbox_inches='tight')

    plt.show()

    # Make plot of sds
for row in range(len(lengths)):
    plt.plot(stds, sd_mean[row,:], label=str(lengths[row]))
    plt.scatter(stds, sd_mean[row,:])
    plt.fill_between(stds, (sd_mean[row,:] - sd_std[row,:]),
                         (sd_mean[row,:] + sd_std[row,:]), alpha=.1)
    plt.ylim((-0.3, 1.1))
    plt.xlabel("Standard deviation of noise")
    plt.ylabel("Correlation coefficient")
    plt.title("Performance on test data \n Interval = 5")
    plt.legend(title = "Number of obs", loc='center left', bbox_to_anchor=(1, 0.5))

path = "C:/Users/5605407/Documents/PhD/PythonProjects/timeseriesanalysis/results/figures/" + path_name + "/test/"

if not os.path.exists(path):
    os.makedirs(path)

plt.savefig(path + "/diff per std", bbox_inches='tight')

plt.show()
