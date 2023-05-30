"""packages used in this file"""
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import RecurrencePlot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_time_series(time_series, filename = ""):
    """
    Plots the time series. Saves plot if a filename is given.
    """
    plt.figure(figsize=(5, 5))
    plt.plot(time_series)
    plt.title("Time Series Plot", fontsize = 18)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/time_series_plot_" + filename)

    plt.show()

def plot_recurrence(time_series, delay = 1, eps = 0.05, filename = ""):
    """
    :type eps: double
    :type delay: integer
    :param time_series: a np.array time series with uniform observation intervals
    :param time_points: observation times of time series
    :param delay: time gap between two back-to-back points of the time series
    :param eps: threshold for the maximum distance to be considered recurrent
    :param filename: name under which the recurrence plot is saved
    :return: a recurrence plot

    Shows a recurrence plot of time_series with given delay and threshold epsilon.
    Saves plot if a filename is provided.
    """
    # transform time series into np.array
    short_time_series = np.array([time_series])

    # initiate a recurrence plot
    try:
        recurrence_info = RecurrencePlot(time_delay=delay, threshold=eps)

        # transform time series into a recurrence plot
        my_recurrence_plot = recurrence_info.transform(short_time_series)

        # plot figure
        plt.figure(figsize=(5, 5))
        plt.imshow(my_recurrence_plot[0], cmap="binary", origin="lower")
        plt.title("Recurrence Plot", fontsize=18)
        plt.tight_layout()

        # save plot iff filename is provided
        if filename != "":
            plt.savefig("../results/figures/recurrence_plot_" + filename)

        # show plot
        plt.show()

    except:
        print("An exception occurred. Are you sure time_delay is an integer"
              " and threshold a double?")

def plot_autocorrelation(time_series, filename = ""):
    """
    Plots the autocorrelation function of the time series.
    Saves plot if a filename is given.
    """
    # Determine which lags to show on the x-axis
    if (len(time_series) > 50):
        lags = np.arange(0, len(time_series), step = len(time_series) / 50)
    else:
        lags = np.arange(0, len(time_series))

    plt.figure(figsize=(5, 5))
    plot_acf(time_series, lags = np.array(lags))
    plt.title("Autocorrelation Plot", fontsize = 18)
    plt.ylim(-1.1, 1.1)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/plot_acf_" + filename)

    plt.show()

def plot_partial_autocorrelation(time_series, filename = ""):
    """
        Plots the partial autocorrelation function of the time series.
        Saves plot if a filename is given.
        """
    # Determine which lags to show on the x-axis
    if (len(time_series) > 50):
        lags = np.arange(0, len(time_series), step=len(time_series) / 50)
    else:
        lags = np.arange(0, len(time_series))

    plt.figure(figsize=(5, 5))
    plot_pacf(time_series, lags=np.array(lags))
    plt.title("Partial Autocorrelation Plot", fontsize=18)
    plt.ylim(-1.1, 1.1)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/plot_pacf_" + filename)

    plt.show()

def plot_3D(filename = ""):
    return 0

if __name__ == "__main__":
    # Create a toy time series using the sine function
    time_points = np.linspace(0, 4 * np.pi, 1000)
    time_series_x = np.sin(time_points)

    # Make plots (time series, recurrence and ACP)
    plot_time_series(time_series_x, filename="sin")
    plot_recurrence(time_series_x, delay = 1.575, eps = np.pi/18, filename = "sin")
    plot_autocorrelation(time_series_x, filename="sin")
