from matplotlib.ticker import MaxNLocator
from create_dummy_time_series import *


def create_hankel_matrix(ts, lag=1, E=2):
    """
    Returns the first E+1 rows of the Hankel-matrix of a time series. Each consecutive row contains
    the time series shifted backwards lag time steps.
    """
    hankel_matrix = []

    for i in range(E + 1):
        if i == 0:
            # Add original time series
            delayed_ts = ts[(E - i) * lag:]
        else:
            # Add time series that is shifted i times
            delayed_ts = ts[(E - i) * lag:-i * lag]
        hankel_matrix.append(delayed_ts)

    # turn list into np.array
    hankel_matrix = np.stack(hankel_matrix, axis=0)

    return hankel_matrix


def shift_hankel_matrix(matrix, horizon):
    # Shift first row to the left horizon-1 times

    if horizon > 1:
        one_step_ahead = matrix[0, :]
        n_step_ahead = one_step_ahead[horizon - 1:]
        matrix = matrix[:, :-(horizon-1)]
        matrix[0, :] = n_step_ahead

    return matrix


def hankel_to_lib(matrix):
    """From a hankel matrix, creates a library of input,output-pairs"""
    lib = []

    # For each column, create a tuple (time-delay vector, prediction)
    for col in range(matrix.shape[1]):
        tuple = [matrix[1:, col], matrix[0, col]]
        lib.append(tuple)

    return lib


def create_distance_matrix(X, Y):
    """
    Returns a matrix of distances between time-delayed embedding vectors in Y to vectors in X
    """
    dist_matrix = np.zeros((len(X), len(Y)))

    for p in range(len(X)):
        for q in range(len(Y)):
            x = X[p][0]
            y = Y[q][0]
            dist = np.linalg.norm((y - x))
            dist_matrix[p, q] = dist

    return dist_matrix


def embed_time_series(ts, lag=1, E=2, horizon=1):

    #TODO:
    # if multiple time series are input
    # # make sure time_series is a list of np_arrays
    # if isinstance(ts, list):
    #     if isinstance(ts[0], float) or isinstance(ts[0], int):
    #         ts = np.array(ts)
    #         ts = [ts]
    #     if isinstance(ts[0], list):
    #         for i in range(len(ts)):
    #             ts[i] = np.array(ts[i])
    # elif isinstance(ts, np.ndarray):
    #     try:
    #         np.shape(ts)[1]
    #     except:
    #         ts = np.reshape(ts, (1,len(ts)))

    #for series in range(len(ts)):

    hankel_matrix = create_hankel_matrix(ts, lag, E)
    hankel_matrix = shift_hankel_matrix(hankel_matrix, horizon)
    lib = hankel_to_lib(hankel_matrix)

    return lib


def split_train_test(lib, percentage_train):

    if percentage_train < 10 or percentage_train > 90:
        percentage_train = 60

    # Split predictor from response variables
    X, y = [], []
    for point in lib:
        X.append(point[0])
        y.append(point[1])

    # Split into training and test set (time ordered)
    cut_off = int(len(X) * percentage_train/100.0)
    X_train, y_train = X[:cut_off], y[:cut_off]
    X_test, y_test = X[cut_off + 1:], y[cut_off + 1:]


    return X_train, y_train, X_test, y_test


def split_rolling_base(lib, n_bins=4, min_tr_size=1):

    # Initialize training and test sets
    X_trains, y_trains = [], []
    X_tests, y_tests = [], []

    # Split predictor variables from target variables
    X, y = [], []
    for point in lib:
        X.append(point[0])
        y.append(point[1])

    # Determine bin sizes
    bin_size = max(1, int((len(lib)-min_tr_size) / n_bins))

    if n_bins > int((len(lib) - min_tr_size)/bin_size):
        n_bins = int((len(lib) - min_tr_size)/bin_size)
        #print("Library is not large enough for given number of bases in rolling base CV. "
              #"Changing to " + str(n_bins))

    stop = len(lib)

    # For each bin, fill a training and test set
    for i in range(n_bins):
        start = stop - bin_size

        X_test, y_test = X[start:stop], y[start:stop]
        X_train, y_train = X[0:start], y[0:start]

        X_trains.append(X_train)
        X_tests.append(X_test)
        y_trains.append(y_train)
        y_tests.append(y_test)

        stop = stop - bin_size

    return X_trains, y_trains, X_tests, y_tests


def transform_lib(lib):
    X, y = [], []
    for point in lib:
        X.append(point[0])
        y.append(point[1])
    return X, y


def plot_results(results, method):

    # Performance measures per E or theta
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(method + "\n Performance measures")

    x = np.arange(1, len(results['corr_list'])+1)

    axs[0].plot(x, results['corr_list'])
    axs[0].scatter(x, results['corr_list'])
    axs[0].set_ylabel("rho")

    axs[1].plot(x, results['mae_list'])
    axs[1].scatter(x, results['mae_list'])
    axs[1].set_ylabel("MAE")

    axs[2].plot(x, results['rmse_list'])
    axs[2].scatter(x, results['rmse_list'])
    axs[2].set_ylabel("RMSE")

    for i in range(1, len(results['corr_list']) + 1):
        axs[0].axvline(x=i, linestyle='--', color='grey', alpha=0.4)
        axs[1].axvline(x=i, linestyle='--', color='grey', alpha=0.4)
        axs[2].axvline(x=i, linestyle='--', color='grey', alpha=0.4)

    if method == "simplex":
        fig.supxlabel("E")
    else:
        fig.supxlabel("theta")

    axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.show()

    # Observed vs predictions
    fig2, axs2 = plt.subplots()

    axs2.scatter(results['observed'], results['predicted'])
    min_ = min([min(results['observed']), min(results['predicted'])])
    max_ = max([max(results['observed']), max(results['predicted'])])
    axs2.plot([min_, max_], [min_, max_])
    fig2.suptitle(method + "\n Observed vs Predicted")
    axs2.set_xlabel("Observed")
    axs2.set_ylabel("Predicted")
    #plt.show()

    fig2.show()

    return 0