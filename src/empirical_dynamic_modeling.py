import math

import matplotlib.pyplot as plt
from pyEDM import *
from scipy.stats import pearsonr
from preprocessing import *
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
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
            ts = np.reshape(ts, (1,len(ts)))


    for series in range(len(ts)):
        hankel_matrix = create_hankel_matrix(ts[series], lag, E)

        # For each column, create a tuple (time-delay vector, prediction)
        for col in range(hankel_matrix.shape[1]):
            tuple = [hankel_matrix[1:, col], hankel_matrix[0, col]]
            lib.append(tuple)

    return lib


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
    results['rmse'] = math.sqrt(mean(np.square(np.subtract(results['observed'],results['predicted']))))

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


def simplex_projection(ts, lag, max_E = 10, val_method ="LB", val_int = None):

    if val_method == "LB" and val_int == None:
        val_int = 50

    results = dict()
    results['corr'], results['E'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    E = 1
    while E < max_E + 1:
        lib = embed_time_series(ts, lag, E)

        # make sure max_E is not too large
        if max_E > np.sqrt(len(lib)):
            max_E = int(np.floor(np.sqrt(len(lib))))
            print("max_E too large. Changing to " + str(max_E) + ".")

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

        E += 1

    print("Optimal dimension found by Simplex is E = " + str(results['E']) +
          " (phi = " + str(results['corr']) + ")")

    return results


def smap(ts, lag, E, val_method ="LB", val_int = None):

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

    results_simplex = simplex_projection(ts, lag, max_E, validation_method, validation_int)
    plot_results(results_simplex, "simplex")

    results_smap = smap(ts, lag, results_simplex['E'], validation_method, validation_int)
    plot_results(results_smap, "s-map")

    print("Optimal theta found by S-map is " + str(results_smap['theta']) + " (phi = " + str(results_smap['corr']) + " )")

    return results_smap


if __name__ == "__main__":
    # Set parameters
    E = 5


    # Simulate the Lorenz System
    x_, y_, z_, t_ = simulate_lorenz(tmax=80, ntimesteps=3000, obs_noise_sd=0)


    # Change sampling interval
    spin_off = 300
    sampling_interval = 20

    x = [x_[i] for i in range(1, len(x_)) if i % sampling_interval == 0 and i > spin_off]
    t = [t_[i] for i in range(1, len(x_)) if i % sampling_interval == 0 and i > spin_off]


    # Embed time series
    lib = embed_time_series(x, lag=1, E=E)


    # Split into training and test set (time ordered)
    cut_off = int(len(lib) * 0.7)
    X_train = x[:cut_off]
    X_test = x[cut_off + 1:]


    # Plot time series
    plt.plot(t_, x_, color='grey', linewidth=1, linestyle='--')
    plt.plot(t, x, color='orange', linewidth=2)
    plt.scatter(t, x, color='red')
    plt.axvline(x=t[0], color='red')
    plt.axvline(x=((t[cut_off] + t[cut_off + 1]) / 2.0), color='red')
    plt.title('Input time series')
    #plt.show()


    # Print information
    print("Number of training samples: " + str(len(X_train)))
    print("Number of test samples: " + str(len(X_test)))
    print("Sampling interval: " + str(t[1] - t[0]))


    results = edm(x, lag=1, max_E=E, validation_method="LB", validation_int=70)


    # Plot error over test set
    fig, axs = plt.subplots(2, sharex=True)

    ax1 = axs[0]
    ax2 = axs[1]

    ax1.set_xlabel('time')
    ax1.set_ylabel('x(t) and pred_x(t)')
    ax1.plot(t[cut_off+1:], x[cut_off+1:], color='grey', linewidth=1, linestyle='--')
    ax1.scatter(t[-len(results['predicted']):], results['predicted'], color='tab:cyan')
    ax1.fill_between(t[-len(results['predicted']):], results['predicted'], results['observed'], color='tab:cyan', alpha=0.3)


    ax2.set_ylabel('Absolute error')
    plt.plot(t[-len(results['predicted']):], np.abs(np.array(results['predicted'])-np.array(results['observed'])), color='tab:cyan')


    fig.tight_layout()
    plt.show()



    #TODO: for cross validation, do not calculate CORR, RMSE, MAE first and then aggregate, but collect all observations
    # and predictions, then calculate these measures. Otherwise LOO-CV doesn't work.