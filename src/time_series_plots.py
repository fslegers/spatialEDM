"""packages used in this file"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyts.image import RecurrencePlot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf
from mayavi import mlab

def plot_time_series(time_series, obs_times = None, filename = ""):
    """
    Plots the time series. Time series can be a single time series or an array of time series.
    Saves plot if a filename is given.
    :param time_series: a single or an array of time series.
    :param filename: if given, the time series plot is saved using this name.
    """

    #check if we are dealing with one or multiple time series
    if len(np.shape(time_series)) == 1 or np.shape(time_series)[1] == 1:
        # if no observation times are given, create array
        if obs_times == None:
            obs_times = np.arange(0, len(time_series))

        plt.plot(obs_times, time_series)
        plt.xlabel('t')
        plt.ylabel('x')
        if(filename != ""):
            plt.suptitle(filename, fontsize = 18)
        plt.title("Time series Plot")

    else:
        # if no observation times are given, create array
        if obs_times == None:
            obs_times = np.arange(0, np.shape(time_series)[0])

        fig, axs = plt.subplots(np.shape(time_series)[1])
        fig.suptitle("Time Series Plots", fontsize = 18)
        for i in range(np.shape(time_series)[1]):
            axs[i].plot(obs_times, time_series[:,i])
            axs[i].set(ylabel = "x" + str(i), xlabel = "t")

        plt.suptitle((str(filename)+"\n Time series plot"), fontsize = 15)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/time_series_plot_" + filename)

    plt.show()

def plot_embedding(time_series, E = 3, lag = 1, filename = ""):
    if(E <= 1 or E > 3):
        print("Cannot plot embedding of dimension ", str(E))
        return 0

    if(E == 2):
        x = time_series
        y = np.roll(x, -lag)

        plt.plot(x[:-lag], y[:-lag])
        plt.show()

    else:
        x = time_series
        y = np.roll(time_series, -lag)
        z = np.roll(time_series, -2*lag)

        mlab.figure(bgcolor = (1,1,1), fgcolor = (0,0,0))
        mlab.plot3d(x[:-2*lag], y[:-2*lag], z[:-2*lag], np.arange(0, len(x[:-2*lag]),1),
                    colormap = "PuRd", tube_radius = 0.1)
        mlab.axes(xlabel = "x(t)", ylabel = "x(t-"+str(lag)+")", zlabel = "x(t-"+str(2*lag)+")")

        mlab.show()

        mlab.savefig(filename = "plot_embedding_" + filename)

        mlab.close()





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

#TODO
#Joint Recurrence Plot

def plot_autocorrelation(time_series, filename = ""):
    """
    Plots the autocorrelation function of the time series.
    Saves plot if a filename is given.
    Prints for which lag the autocorrelation first crosses the x-axis.
    """
    # Determine which lags to show on the x-axis
    n_col = np.shape(time_series)[0]

    if n_col > 50:
        lags = np.arange(0, np.floor(.5*len(time_series)), step = n_col / 50)
    else:
        lags = np.arange(0, np.floor(.5*len(time_series)))

    plt.figure(figsize=(5, 5))
    plot_acf(time_series, lags = np.array(lags))
    plt.title("Autocorrelation Plot", fontsize = 18)
    plt.ylim(-1.1, 1.1)

    # Find first switch from positive to negative
    auto_correlations = acf(time_series, nlags=len(time_series), bartlett_confint=True)

    i = 1
    while(i < len(time_series)):
        if auto_correlations[i] <= 0:
            print("first negative value at lag ", i)
            break
        i += 1

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
    if len(time_series) > 50:
        lags = np.arange(0, np.floor(0.5*len(time_series)), step=len(time_series) / 50)
    else:
        lags = np.arange(0, np.floor(0.5*len(time_series)))

    plt.figure(figsize=(5, 5))
    plot_pacf(time_series, lags=np.array(lags))
    plt.title("Partial Autocorrelation Plot", fontsize=18)
    plt.ylim(-1.1, 1.1)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/plot_pacf_" + filename)

    plt.show()

def plot_correlation(x, y, window_size = 0, filename = ""):
    """
    Plots Pearsons correlation coefficient within a moving window over time.
    :param x: time series
    :param y: time series
    :param window_size: how many observations are used per correlation coefficient
    """
    t_max = np.shape(x)[0]

    # Choose size of sliding window for which correlations are calculated
    if window_size == 0:
        window_size = int(np.ceil(t_max/10))

    counter = 0
    time_points = []
    pearson_correlations = []
    while counter < (t_max - window_size):
        x_subset = x[counter:(counter+window_size),]
        y_subset = y[counter:(counter+window_size),]
        pearson_correlations.append(stats.pearsonr(x_subset, y_subset)[0])
        time_points.append(counter + window_size)
        counter += 1

    plt.plot(time_points, pearson_correlations)
    plt.title("Correlation, time-plot")
    plt.suptitle(filename, fontsize = 18)
    plt.ylim(-1.01, 1.01)

    # save plot iff filename is provided
    if filename != "":
        plt.savefig("../results/figures/plot_correlation_" + filename)

    plt.show()

def make_lag_scatterplot(time_series, lag, filename = ""):
    x = time_series[lag:]
    y = time_series[:-lag]

    plt.plot(y, x)
    plt.xlabel("x(t - tau)")
    plt.ylabel("x(t)")
    plt.title("Lag plot, tau = " + str(lag), fontsize = 18)

    if(filename != ""):
        plt.savefig('scatterplot_'+filename)

    plt.show()

def make_3d_plot(x, y, z, filename = "", tube_radius = 0.1, colors = "PuRd"):
    """
    Creates (and saves) a 3D plot of time series of variables x, y and z.
    :param x: time series (array) of variable x
    :param y: time series (array) of variable y
    :param z: time series (array) of variable z
    :param filename: if filename if given, the plot will be saved under this name.
    """
    times = np.arange(0,len(x)) # for colormap of trajectory

    mlab.figure(bgcolor=(0,0,0)) # set black background
    mlab.plot3d(x, y, z, times, tube_radius = tube_radius, colormap = colors)

    # save plot iff filename is provided
    if filename != "":
        mlab.savefig("../results/figures/plot_3D_" + filename + ".png")

    mlab.show()

def example_mayavi():
    n_mer, n_long = 6, 11
    dphi = np.pi / 1000.0
    phi = np.arange(0.0, 2 * np.pi + 0.5 * dphi, dphi)
    mu = phi * n_mer
    x_ = np.cos(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    y_ = np.sin(mu) * (1 + np.cos(n_long * mu / n_mer) * 0.5)
    z_ = np.sin(n_long * mu / n_mer) * 0.5

    mlab.plot3d(x_, y_, z_, np.sin(mu), tube_radius=0.025, colormap='Spectral')
    mlab.show()

if __name__ == "__main__":
    #example_mayavi()

    x = np.arange(0, 20, 0.1)
    x = np.sin(x)


    plot_embedding(x, 3, 3)
