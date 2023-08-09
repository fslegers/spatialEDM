import math
import itertools
import matplotlib.cm
from pyEDM import *
from scipy.stats import pearsonr
from preprocessing import *
from src.create_dummy_time_series import simulate_lorenz
from src.time_series_plots import plot_time_series
import random

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import r2_score

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


def embed_time_series(ts, lag=1, E=2):
    lib = []

    # make sure time_series is a list of np_arrays
    if isinstance(ts, list):
        if isinstance(ts[0], float) or isinstance(ts[0], int):
            ts = np.array(ts)
            ts = [ts]
        if isinstance(ts[0], list):
            for i in range(len(ts)):
                ts[i] = np.array(ts[i])
    elif isinstance(ts, np.ndarray):
        try:
            np.shape(ts)[1]
        except:
            ts = np.reshape(ts, (1,100))


    for series in range(len(ts)):
        hankel_matrix = create_hankel_matrix(ts[series], lag, E)

        # For each column, create a tuple (time-delay vector, prediction)
        for col in range(hankel_matrix.shape[1]):
            tuple = [hankel_matrix[1:, col], hankel_matrix[0, col]]
            lib.append(tuple)

    return lib


def simplex_forecasting(training_set, test_set, lag, dim):

    # Calculate distances from test points to training points
    dist_matrix = create_distance_matrix(test_set, training_set)

    observed = []
    predicted = []

    for target in range(len(test_set)):
        # Select E + 1 nearest neighbors
        dist_to_target = dist_matrix[target,:]
        nearest_neighbors = np.argpartition(dist_to_target, (0, dim+2))
        nearest_neighbors = np.arange(len(training_set))[nearest_neighbors[0:dim+1]]
        min_distance = dist_to_target[nearest_neighbors[0]]

        weighted_average = 0
        total_weight = 0
        weights = []

        # Calculate weighted sum of next value
        for neighbor in nearest_neighbors:
           if min_distance == 0:
               if dist_to_target[neighbor] == 0:
                   weight = 1
               else:
                   weight = 0.000001
           else:
               weight = np.exp(-dist_to_target[neighbor] / min_distance)

           next_val = training_set[neighbor][1]
           weighted_average += next_val * weight
           total_weight += weight
           weights.append(weight)

        # Calculate weighted average
        weighted_average = weighted_average / total_weight
        predicted.append(weighted_average)
        observed.append(test_set[target][1])

    results = dict()

    # Calculate performance measures
    results['corr'] = pearsonr(observed, predicted)[0]
    results['mae'] = mean(abs(np.subtract(observed, predicted)))
    results['rmse'] = math.sqrt(mean(np.square(np.subtract(observed, predicted))))

    results['observed'] = observed
    results['predicted'] = predicted

    return results


def smap_forecasting(training_set, test_set, theta):
    # Calculate distances from test points to training points
    dist_matrix = create_distance_matrix(test_set, training_set)

    results = dict()
    results['observed'], results['predicted'] = [], []
    results['theta'] = theta

    for target in range(len(test_set)):
        # Calculate weights for each training point
        distances = dist_matrix[target, :]
        weights = np.exp(-theta * distances / mean(distances))

        # Fill vector of weighted future values of training set
        next_values = [point[1] for point in training_set]
        b = np.multiply(weights, next_values)

        # Fill matrix A
        prev_values = np.array([item[0] for item in training_set])
        A = prev_values * np.array(weights)[:, None]

        # Calculate coefficients C using the pseudo-inverse of A (via SVD)
        coeffs = np.matmul(np.linalg.pinv(A), b)

        # Make prediction
        prediction = np.matmul(np.array(test_set[target][0]), coeffs)
        results['observed'].append(test_set[target][1])
        results['predicted'].append(prediction)

    # Calculate performance measures
    results['corr'] = pearsonr(results['observed'], results['predicted'])[0]
    results['mae'] = mean(abs(np.subtract(results['observed'], results['predicted'])))
    results['rmse'] = math.sqrt(mean(np.square(np.subtract(results['observed'], results['predicted']))))

    return results


def forecasting_for_cv(lib, k, lag, E, method):
    """
    :param k: the number of groups that the (concatenated) time series is to be split into
    """
    if k > len(lib):
        print("number of folds is too large")
        k = len(lib)

    results = dict()
    results['corr'], results['mae'], results['rmse'] = 0, 0, 0
    results['observed'], results['predicted'] = [], []
    block_size = int(len(lib) / k)

    for fold in range(k):

        if fold != k - 1:
            # Split into training and test set
            test_set = lib[fold * block_size:(fold + 1) * block_size]
            training_set = lib[:fold * block_size] + lib[(fold + 1) * block_size:]
        else:
            test_set = lib[fold * block_size:]
            training_set = lib[:fold * block_size]

        # Perform forecasting
        if method == "simplex":
            result_fold = simplex_forecasting(training_set, test_set, lag, E)
        else:
            result_fold = smap_forecasting(training_set, test_set, method)

        results['observed'] += result_fold['observed']
        results['predicted'] += result_fold['predicted']

    # Calculate performance measures
    results['corr'] = pearsonr(results['observed'], results['predicted'])[0]
    results['mae'] = mean(abs(np.subtract(results['observed'], results['predicted'])))
    results['rmse'] = math.sqrt(mean(np.square(np.subtract(results['observed'], results['predicted']))))

    return results


def simplex_loop(ts, lag, max_E = 10, val_method ="LB", val_int = None):

    if val_method == "LB" and val_int == None:
        val_int = 50

    results = dict()
    results['corr'], results['E'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    for E in range(1, max_E + 1):
        lib = embed_time_series(ts, lag, E)

        if val_method == "CV": # k-fold cross validation
            result_E = forecasting_for_cv(lib, val_int, lag, E, method="simplex")

        else: # last block validation
            split = int(np.ceil(val_int/100 * len(lib)))
            training_set = lib[:split]
            test_set = lib[split:]
            result_E = simplex_forecasting(training_set, test_set, lag, E)

        results['corr_list'].append(result_E['corr'])
        results['mae_list'].append(result_E['mae'])
        results['rmse_list'].append(result_E['rmse'])

        if result_E['corr'] > results['corr'] or E == 1:
            results['corr'] = result_E['corr']
            results['E'] = E
            results['observed'] = result_E['observed']
            results['predicted'] = result_E['predicted']

    print("Optimal dimension found by Simplex is E = " + str(results['E']) + " (phi = " + str(results['corr']) + ")")

    return results


def smap_loop(ts, lag, E, val_method ="LB", val_int = None):

    if val_method == "LB" and val_int == None:
        val_int = 50

    results = dict()
    results['corr'], results['theta'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    lib = embed_time_series(ts, lag, E)

    # last block validation
    if val_method != "CV":
        split = int(np.ceil(val_int / 100 * len(lib)))
        training_set = lib[:split]
        test_set = lib[split:]

    for theta in range(11):

        if val_method == "CV":
            result_theta = forecasting_for_cv(lib, val_int, lag, E, theta)
        else:
            result_theta = smap_forecasting(training_set, test_set, theta)

        # Update optimal predictions
        if result_theta['corr'] > results['corr'] or theta == 1:
            results['theta'] = theta
            results['corr'] = result_theta['corr']
            results['mae'] = result_theta['mae']
            results['rmse'] = result_theta['rmse']
            results['observed'] = result_theta['observed']
            results['predicted'] = result_theta['predicted']

        results['corr_list'].append(result_theta['corr'])
        results['mae_list'].append(result_theta['mae'])
        results['rmse_list'].append(result_theta['rmse'])


    return results


def edm(ts, lag, max_E, validation_method, validation_int):
    results_simplex = simplex_loop(ts, lag, max_E, validation_method, validation_int)
    results_smap = smap_loop(ts, lag, results_simplex['E'], validation_method, validation_int)
    return results_smap


def kernel(x, y, phi, tau, r):
    """
    As described by Munch et al. 20217
    :param x: embedding vector 1
    :param y: embedding vector 2
    :param phi: controls the wiggliness of f in the direction of each time-lag
    :param tau: controls the prior variance in f at a given point
    :param L: dimension
    :param r: max(y) - min(y)
    :return: covariance between f(x) and f(y)
    """
    prod = squared_exponential(phi[0] * abs(x[0] - y[0]) / r)
    for i in range(1, len(x)):
        prod *= squared_exponential(phi[i] * abs(x[i] - y[i]) / r)
    return tau**2 * prod


def squared_exponential(d):
    return exp(-d**2)


def gpr_forecasting(lib, X_train, y_train, X_test, y_test):

    kernel = 1 * RBF(length_scale = 1.0, length_scale_bounds=(1e-2, 1e2))
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    model.fit(X_train, y_train)

    y_pred_te, y_pred_te_std = model.predict(X_test, return_std=True)

    results = dict()
    results['observed'] = y_test
    results['predicted'] = y_pred_te
    results['std'] = y_pred_te_std
    results['corr'] = pearsonr(results['observed'], results['predicted'])[0]
    results['mae'] = mean(abs(np.subtract(results['observed'], results['predicted'])))
    results['rmse'] = math.sqrt(mean(np.square(np.subtract(results['observed'], results['predicted']))))

    return results


def gpr(ts, lag, E, val_method, val_int = None):

    if val_method == "LB" and val_int == None:
        val_int = 50

    lib = embed_time_series(ts, lag, E)

    if val_method == "CV":
        #result = forecasting_for_cv(lib, val_int, lag, E, method="gpr")
        #TODO: implement method="GPR" in forecasting_for_cv
        print("k-fold cross validation")

    else:
        split = int(np.ceil(val_int / 100 * len(lib)))
        training_set = lib[:split]
        test_set = lib[split:]

        X_train = np.array([item[0] for item in training_set])
        X_test = np.array([item[0] for item in test_set])
        y_train = np.array([item[1] for item in training_set])
        y_test = np.array([item[1] for item in test_set])

        result = gpr_forecasting(lib, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

        lim_min = min(min(result['observed']), min(result['predicted'])) - 0.1
        lim_max = max(max(result['observed']), max(result['predicted'])) + 0.1

        for target in range(len(result['predicted'])):
            x = result['observed'][target]
            y_min = result['predicted'][target] - result['std'][target]
            y_max = result['predicted'][target] + result['std'][target]
            plt.plot((x, x), (y_min, y_max), linewidth=4, color='b', alpha=.15)

        plt.plot([lim_min, lim_max], [lim_min, lim_max], linewidth=1, color='black', linestyle='--')
        plt.scatter(result['observed'], result['predicted'])

        plt.xlim((lim_min,lim_max))
        plt.ylim((lim_min, lim_max))
        plt.xlabel('observed')
        plt.ylabel('predicted')
        plt.title("Forecasting performance GPR")
        plt.show()

    return 0


if __name__ == "__main__":

    a = np.sin(np.arange(1, 1000, 5)/100.0)
    a = [item + random.random()/10.0 for item in a]

    plt.plot(np.arange(len(a)), a)
    plt.scatter(np.arange(len(a)), a)
    plt.title("time series plot")
    plt.show()

    gpr(a, 1, 5, "LB", 50)

    # smap_results = smap_loop(a, lag=1, E=2, val_method="CV", val_int=20)
    #
    # plt.plot(np.arange(0,11), smap_results['corr_list'])
    # plt.scatter(np.arange(0, 11), smap_results['corr_list'])
    # plt.xlabel('theta')
    # plt.ylabel('rho')
    # plt.show()
    #
    # xlim = [min(smap_results['observed']), max(smap_results['observed'])]
    # ylim = [min(smap_results['predicted']), max(smap_results['predicted'])]
    # plt.plot(xlim, ylim, color='black', linewidth=1, linestyle='--')
    # plt.scatter(smap_results['observed'], smap_results['predicted'])
    # plt.xlabel('observed')
    # plt.ylabel('predicted')
    # plt.title("Forecasting performance (theta = " + str(smap_results['theta'])+")")
    # plt.show()

    #TODO: for cross validation, do not calculate CORR, RMSE, MAE first and then aggregate, but collect all observations
    # and predictions, then calculate these measures. Otherwise LOO-CV doesn't work.