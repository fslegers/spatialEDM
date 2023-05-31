import numpy as np
import sklearn.preprocessing
from matplotlib import pyplot as plt


def summarize_statistics(time_series, window_size = 10, filename = ""):
    """
    Plots a single time series, its moving average and its moving variance.
    Can be used to detect nonstationarity.
    """
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

def normalize_time_series(time_series):
    """
    Standardizes each series in time_series.
    :param time_series: ndarray of a single or multiple time series.
    :return standardized time series with mean 0 and unit variance
    """

    scaler = sklearn.preprocessing.StandardScaler().fit(time_series)
    time_series_scaled = scaler.transform(time_series)

    print(time_series_scaled.mean(axis = 0))
    print(time_series_scaled.std(axis = 0))

    return(time_series_scaled)


if __name__ == "__main__":
    x = np.arange(0,10,0.1)
    y = np.sin(x)

    summarize_statistics(x, window_size=20)
    summarize_statistics(y, window_size=20)

    x_normalized = normalize_time_series(np.array([x]).T)

    xy_normalized = normalize_time_series(np.array([x,y]))
