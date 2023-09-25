import math
import pandas as pd

from matplotlib.ticker import MaxNLocator
from pyEDM import *
from scipy.stats import pearsonr

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


def simplex_forecasting(X_train, y_train, X_test, y_test, dim):

    # Calculate distances from test points to training points
    dist_matrix = create_distance_matrix(X_test, X_train)


    observed = []
    predicted = []


    for target in range(len(X_test)):

        # Select E + 1 nearest neighbors
        dist_to_target = dist_matrix[target,:]
        nearest_neighbors = np.argpartition(dist_to_target, (0, dim+2))
        nearest_neighbors = np.arange(len(X_train))[nearest_neighbors[0:dim+1]]
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


            next_val = y_train[neighbor]
            weighted_average += next_val * weight
            total_weight += weight
            weights.append(weight)


        # Calculate weighted average
        weighted_average = weighted_average / total_weight
        predicted.append(weighted_average)
        observed.append(y_test[target])


    results = dict()


    # Calculate performance measures
    results['corr'] = pearsonr(observed, predicted)[0]
    results['mae'] = mean(abs(np.subtract(observed, predicted)))
    results['rmse'] = math.sqrt(mean(np.square(np.subtract(observed, predicted))))
    results['observed'] = observed
    results['predicted'] = predicted


    return results


def smap_forecasting(X_training, y_training, X_test, y_test, theta):

    # Calculate distances from test points to training points
    dist_matrix = create_distance_matrix(X_test, X_training)


    results = dict()
    results['observed'], results['predicted'] = [], []
    results['theta'] = theta


    for target in range(len(X_test)):


        # Calculate weights for each training point
        distances = dist_matrix[target, :]
        weights = np.exp(-theta * distances / mean(distances))


        # Fill vector of weighted future values of training set
        next_values = y_training
        b = np.multiply(weights, next_values)


        # Fill matrix A
        prev_values = np.array(X_training)
        A = prev_values * np.array(weights)[:, None]


        # Calculate coefficients C using the pseudo-inverse of A (via SVD)
        coeffs = np.matmul(np.linalg.pinv(A), b)


        # Make prediction
        prediction = np.matmul(np.array(X_test[target]), coeffs)
        results['observed'].append(y_test[target])
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
    #TODO: Implement using scikit-learn cross_validation
    return(0)
    # if k > len(lib):
    #     print("number of folds is too large")
    #     k = len(lib)
    #
    # results = dict()
    # results['corr'], results['mae'], results['rmse'] = 0, 0, 0
    # results['observed'], results['predicted'] = [], []
    # block_size = int(len(lib) / k)
    #
    # for fold in range(k):
    #
    #     if fold != k - 1:
    #         # Split into training and test set
    #         test_set = lib[fold * block_size:(fold + 1) * block_size]
    #         training_set = lib[:fold * block_size] + lib[(fold + 1) * block_size:]
    #     else:
    #         test_set = lib[fold * block_size:]
    #         training_set = lib[:fold * block_size]
    #
    #     # Perform forecasting
    #     if method == "simplex":
    #         result_fold = simplex_forecasting(training_set, test_set, lag, E)
    #     else:
    #         result_fold = smap_forecasting(training_set, test_set, method)
    #
    #     results['observed'] += result_fold['observed']
    #     results['predicted'] += result_fold['predicted']
    #
    # # Calculate performance measures
    # results['corr'] = pearsonr(results['observed'], results['predicted'])[0]
    # results['mae'] = mean(abs(np.subtract(results['observed'], results['predicted'])))
    # results['rmse'] = math.sqrt(mean(np.square(np.subtract(results['observed'], results['predicted']))))
    #
    # return results


def simplex_projection(ts, lag, max_E = 10, plotting=True):

    results = dict()
    results['corr'], results['E'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    E = 1

    if max_E > np.sqrt(len(ts)):
        max_E = np.floor(np.sqrt(len(ts)))
        print("max_E too large. Changing to " + str(max_E))


    while E < max_E + 1:

        lib = embed_time_series(ts, lag, E)
        X_train, y_train, X_test, y_test = split_train_test(lib, 70)


        result_E = simplex_forecasting(X_train, y_train, X_test, y_test, E)


        results['corr_list'].append(result_E['corr'])
        results['mae_list'].append(result_E['mae'])
        results['rmse_list'].append(result_E['rmse'])

        if result_E['corr'] > results['corr'] or E == 1:
            results['corr'] = result_E['corr']
            results['E'] = E
            results['observed'] = result_E['observed']
            results['predicted'] = result_E['predicted']

        E += 1

    if plotting:
        print("Optimal dimension found by Simplex is E = " + str(results['E']) +
          " (phi = " + str(results['corr']) + ")")

    return results


def smap(X_train, y_train, X_test, y_test):


    results = dict()
    results['corr'], results['theta'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    for theta in range(11):

        result_theta = smap_forecasting(X_train, y_train, X_test, y_test, theta)

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


def edm(ts, lag, max_E, plotting=True):

    results_simplex = simplex_projection(ts, lag, max_E, plotting=False)

    if plotting:
        plot_results(results_simplex, "simplex")

    # Embed time series and create training and test sets
    lib = embed_time_series(ts, E=results_simplex['E'])
    X_train, y_train, X_test, y_test = split_train_test(lib, 70)

    results_smap = smap(X_train, y_train, X_test, y_test)

    if plotting:
        plot_results(results_smap, "s-map")

    print("Optimal theta found by S-map is " + str(results_smap['theta'])
        + r"\rho = " + str(results_smap['corr']) + " )")


    return results_smap


def mega_loop(x, t):

    interval_list = np.arange(1, 30, 5)
    t_max_list = np.arange(15, 500, 25)

    error = np.empty((len(interval_list), len(t_max_list)))

    for interval in range(len(interval_list)):
        for t_max in range(len(t_max_list)):

            # Take sample(s) from lorenz trajectory
            n_points = t_max_list[t_max]/interval_list[interval]
            x, t = sample_from_ts(x_, t_, spin_off=0, n_points=n_points, sampling_interval=interval_list[interval])

            results = edm(x, lag=1, max_E=10, plotting=False)
            print("rij: " + str(interval) + ", kolom: " + str(n_points) + ", corr: " + str(round(results['corr'],3)))
            error[interval, n_points] = round(results['corr'],5)

    error_df = pd.DataFrame(error,
                            columns=[str(i) for i in t_max_list],
                            index=[str(i) for i in interval_list])








if __name__ == "__main__":

    t_max_lorenz = 500
    resolution_lorenz = 0.01
    n_time_steps = int(t_max_lorenz / resolution_lorenz)

    # Simulate the Lorenz System
    x_, y_, z_, t_ = simulate_lorenz(tmax=t_max_lorenz, ntimesteps=n_time_steps, obs_noise_sd=0)
    plot_time_series(x_, t_, scatter=False)


    mega_loop(x_[100:], t_[100:])


    # # Plot error over test set
    # fig, axs = plt.subplots(2, sharex=True)
    #
    # ax1 = axs[0]
    # ax2 = axs[1]
    #
    # ax1.set_xlabel('time')
    # ax1.set_ylabel('x(t) and pred_x(t)')
    # ax1.plot(t[-len(results['predicted']):], results['observed'], color='grey', linewidth=1, linestyle='--')
    # ax1.plot(t[-len(results['predicted']):], results['predicted'], color='tab:cyan')
    # ax1.fill_between(t[-len(results['predicted']):], results['predicted'], results['observed'], color='tab:cyan', alpha=0.3)
    #
    # ax2.set_ylabel('Absolute error')
    # error = np.abs(np.array(results['predicted']) - np.array(results['observed']))
    # plt.plot(t[-len(results['predicted']):], error, color='tab:cyan')
    #
    # fig.tight_layout()
    # plt.show()


    #TODO: for cross validation, do not calculate CORR, RMSE, MAE first and then aggregate, but collect all observations
    # and predictions, then calculate these measures. Otherwise LOO-CV doesn't work.