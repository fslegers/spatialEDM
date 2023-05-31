import numpy as np
from matplotlib import pyplot as plt


def summarize_statistics(time_series, window_size = 10, filename = ""):
    time_series = time_series.tolist()

    means = []
    variances = []
    counter = 0

    # For each time se
    while(counter < len(time_series) - window_size):
        current_mean = np.mean(time_series[counter : (counter + window_size)])
        current_var = np.var(time_series[counter : (counter + window_size)])
        means.append(current_mean)
        variances.append(current_var)
        counter += 1

    fig, axs = plt.subplots(3)

    axs[0].plot(time_series)
    axs[0].set_title("Time series", fontsize = 10)
    axs[1].plot(means)
    axs[1].set_title("Mean over time", fontsize = 10)
    axs[2].plot(variances)
    axs[2].set_title("Variance over time", fontsize = 10)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/summary_statistics_" + filename)

    plt.show()

if __name__ == "__main__":
    x = np.arange(0,10,0.1)
    y = np.sin(x)

    summarize_statistics(x, window_size=20)
    summarize_statistics(y, window_size=20)
