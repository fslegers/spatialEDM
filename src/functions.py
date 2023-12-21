from src.classes import TimeSeries, EmbeddingVector
from pyEDM import *
from scipy.stats import pearsonr
import math

def create_hankel_matrix(ts_object: TimeSeries, lag, dim):
    """
    Returns the first E+1 rows of the Hankel-matrix of a time series. Each consecutive row contains
    the time series shifted backwards lag time steps. Should only contain one location,species pair!
    """
    # TODO: integrate time stamps into this

    ts = ts_object.points
    hankel_matrix = []

    for i in range(dim + 1):
        if i == 0:
            delayed_ts = ts[(dim - i) * lag:]  # Add original time series
        else:
            delayed_ts = ts[(dim - i) * lag:-i * lag]  # Add time series that is shifted i times
        hankel_matrix.append(delayed_ts)

    hankel_matrix = np.stack(hankel_matrix, axis=0)  # turn list into np.array

    return hankel_matrix


def create_distance_matrix(X, Y):
    """
    Returns a matrix of distances between time-delayed embedding vectors in Y to vectors in X.
    """

    dist_matrix = np.zeros((len(X), len(Y)))

    for p in range(len(X)):
        for q in range(len(Y)):
            x = np.array(X[p].values)
            y = np.array(Y[q].values)
            dist = np.linalg.norm((y - x))
            dist_matrix[p, q] = dist

    return dist_matrix


def shift_hankel_matrix(matrix, horizon):
    # TODO: integrate time stamps into this
    """
    Shift first row to the left horizon-1 times
    """

    if horizon > 1:
        one_step_ahead = matrix[0, :]
        n_step_ahead = one_step_ahead[horizon - 1:]
        matrix = matrix[:, :-(horizon - 1)]
        matrix[0, :] = n_step_ahead

    return matrix


def hankel_to_lib(matrix, location, species):
    """From a hankel matrix, creates a library of input,output-pairs"""
    # TODO: take information on time stamps from matrix

    lib = []
    for col in range(matrix.shape[1]):
        t, spec, loc = matrix[0, col].time_stamp, matrix[0, col].species, matrix[0, col].location
        x = EmbeddingVector([point.value for point in matrix[1:, col]], t, spec, loc)
        y = matrix[0, col]
        lib.append([x, y])
    return lib


def embed_time_series(ts: TimeSeries, lag=1, E=2, horizon=1):
    """returns one library consisting of sub-libraries for each location/species"""
    library = []

    for loc in ts.locations:
        for spec in ts.species:
            sub_ts = TimeSeries([point for point in ts.points if point.location == loc and point.species == spec])
            hankel_matrix = create_hankel_matrix(sub_ts, lag, E)
            hankel_matrix = shift_hankel_matrix(hankel_matrix, horizon)
            lib = hankel_to_lib(hankel_matrix, loc, spec)
            library += lib

    return library


def split_library(lib, cv_method="LB", cv_fraction=0.5):
    if cv_method == "LB":
        X_train, y_train, X_test, y_test = split_last_block(lib, cv_fraction)

    elif cv_method == "RB":
        # TODO: implement
        return 0
        train, test = split_rolling_base(lib, cv_fraction)

    else:
        return 0

    return X_train, y_train, X_test, y_test


def split_last_block(lib, frac):
    # TODO: check order of embedding vectors and points in lib
    # TODO: check if we really need next sentence

    # Split predictor from response variables
    X, y = [], []
    t_min, t_max = math.inf, -math.inf
    for point in lib:
        X.append(point[0])
        y.append(point[1])

        if point[1].time_stamp < t_min:
            t_min = point[1].time_stamp
        if point[1].time_stamp > t_max:
            t_max = point[1].time_stamp

    # Split into training and test set (time ordered)
    cut_off = int(t_min + (t_max - t_min) * frac)  # TODO: naar boven of onder afronden?

    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(X)):
        if y[i].time_stamp <= cut_off:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    return [X_train], [y_train], [X_test], [y_test]


def split_rolling_base(lib, frac):
    # TODO: fix a lot
    min_tr_size = 1
    n_bins = 1

    # Initialize training and test sets
    X_trains, y_trains = [], []
    X_tests, y_tests = [], []

    # Split predictor variables from target variables
    X, y = [], []
    for point in lib:
        X.append(point[0])
        y.append(point[1])

    # Determine bin sizes
    bin_size = max(1, int((len(lib) - min_tr_size) / n_bins))

    # Fix n_bins if necessary
    if n_bins > int((len(lib) - min_tr_size) / bin_size):
        n_bins = int((len(lib) - min_tr_size) / bin_size)

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
    return 0


def initialize_single_result(param, value):
    result = {}
    result['corr'] = 0
    result['mae'] = 0
    result['rmse']  = 0
    result[param] = value
    result['observed'] = []
    result['predicted'] = []

    return result


def update_single_result(result, n_mae, n_corr, obs, pred):
    try:
        diff = np.subtract(obs, pred)
        result['mae'] += mean(abs(diff))
        result['rmse'] += math.sqrt(mean(np.square(diff)))
        result['observed'] += obs
        result['predicted'] += pred
    except:
        n_mae -= 1

    try:
        result['corr'] += pearsonr(obs, pred)[0]
    except:
        n_corr -= 1
    return result, n_mae, n_corr


# def destandardize_result(result, mean, std, diff=[False, 0]):
#     for x in result['predicted']:
#         x = std * x + mean
#
#     if diff[0]:
#         transformed_predictions = []
#         previous_val = diff[1]
#         for y in result['predicted']:
#             transformed_predictions.append(previous_val + y)
#             previous_val += y
#
#     result['predicted'] = transformed_predictions
#
#     return result


def average_result(result, n_mae, n_corr):
    try:
        result['corr'] = result['corr'] / n_corr
    except ZeroDivisionError:
        result['corr'] = None

    try:
        result['mae'] = result['mae'] / n_mae
        result['rmse'] = result['rmse'] / n_mae
    except ZeroDivisionError:
        result['mae'] = None
        result['rmse'] = None
    return result
