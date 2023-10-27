import math
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


def simplex_forecasting(X_trains, y_trains, X_tests, y_tests, dim):

    results = {}
    results['corr'], results['rmse'], results['mae'] = 0, 0, 0
    results['observed'], results['predicted'] = [], []

    for i in range(len(X_trains)):

        X_train, y_train, X_test, y_test = X_trains[i], y_trains[i], X_tests[i], y_tests[i]

        # Calculate distances from test points to training points
        dist_matrix = create_distance_matrix(X_test, X_train)

        obs_this_fold = []
        pred_this_fold = []

        for target in range(len(X_test)):

            # Select E + 1 nearest neighbors
            dist_to_target = dist_matrix[target,:]

            if len(dist_to_target) == dim+1:
                nearest_neighbors=np.arange(0, dim+1)
            else:
                nearest_neighbors = np.argpartition(dist_to_target, min(dim+1, len(dist_to_target)-1))[:dim+1]
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
            pred_this_fold.append(weighted_average)
            obs_this_fold.append(y_test[target])

        diff = np.abs(np.subtract(obs_this_fold, pred_this_fold))
        results['mae'] += mean(diff)
        results['rmse'] += math.sqrt(mean(np.square(diff)))

        try:
            results['corr'] += pearsonr(obs_this_fold, pred_this_fold)[0]
        except:
            results['corr'] += -mean(abs(diff))

        results['observed'] += obs_this_fold
        results['predicted'] += pred_this_fold

    # Calculate performance measures
    results['corr'] = results['corr'] / len(X_trains)
    results['mae'] = results['mae'] / len(X_trains)
    results['rmse'] = results['rmse'] / len(X_trains)

    return results


def smap_forecasting(X_trains, y_trains, X_tests, y_tests, theta):

    results = {}
    results['observed'], results['predicted'] = [], []
    results['corr'], results['rmse'], results['mae'] = 0, 0, 0
    results['theta'] = theta

    for i in range(len(X_trains)):

        X_train, y_train, X_test, y_test = X_trains[i], y_trains[i], X_tests[i], y_tests[i]

        # Calculate distances from test points to training points
        dist_matrix = create_distance_matrix(X_test, X_train)

        obs_this_fold = []
        pred_this_fold = []

        for target in range(len(X_test)):

            # Calculate weights for each training point
            distances = dist_matrix[target, :]
            weights = np.exp(-theta * distances / mean(distances))

            # Fill vector of weighted future values of training set
            next_values = y_train
            b = np.multiply(weights, next_values)

            # Fill matrix A
            prev_values = np.array(X_train)
            A = prev_values * np.array(weights)[:, None]

            # Calculate coefficients C using the pseudo-inverse of A (via SVD)
            coeffs = np.matmul(np.linalg.pinv(A), b)

            # Make prediction
            prediction = np.matmul(np.array(X_test[target]), coeffs)
            obs_this_fold.append(y_test[target])
            pred_this_fold.append(prediction)


        diff = np.subtract(obs_this_fold, pred_this_fold)
        results['mae'] += mean(abs(diff))
        results['rmse'] += math.sqrt(mean(np.square(diff)))
        results['observed'] += obs_this_fold
        results['predicted']+= pred_this_fold

        try:
            results['corr'] += pearsonr(obs_this_fold, pred_this_fold)[0]
        except:
            #print("Cannot calculate pearson correlation.")
            results['corr'] = -mean(abs(diff))

    # Calculate performance measures
    results['corr'] = results['corr']/len(X_trains)
    results['mae'] = results['mae']/len(X_trains)
    results['rmse'] = results['rmse']/len(X_trains)

    return results


def simplex_projection(ts, lag, max_E=10, CV="LB", plotting=True):

    results = {}
    results['corr'], results['E'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    E = 1

    # Check if given max_E is feasible
    if max_E > 0.3*len(ts) - 2:
        max_E = max(1,int(0.3*len(ts)-2))
        #print("max_E too large. Changing to " + str(max_E))

    # Perform KNN-forecasting for each E
    while E < max_E + 1:

        # Embed time series
        lib = embed_time_series(ts, lag, E)

        # Split in training and test set
        if CV == "RBCV":
            X_trains, y_trains, X_tests, y_tests = split_rolling_base(lib, min_tr_size=E+1)
        else:
            X_train, y_train, X_test, y_test = split_train_test(lib, 70)
            X_trains, y_trains, X_tests, y_tests = [X_train], [y_train], [X_test], [y_test]

        result_E = simplex_forecasting(X_trains, y_trains, X_tests, y_tests, E)

        results['corr_list'].append(result_E['corr'])
        results['rmse_list'].append(result_E['rmse'])
        results['mae_list'].append(result_E['mae'])

        if result_E['corr'] > results['corr'] or E == 1:
            results['observed'] = result_E['observed']
            results['predicted'] = result_E['predicted']
            results['corr'] = result_E['corr']
            results['E'] = E

        E += 1

    if plotting:
        print("Optimal dimension found by Simplex is E = " + str(results['E']) +
          r'$\rho$' + " = " + str(results['corr']) + ")")

    return results


def smap(X_train, y_train, X_test, y_test):

    results = {}
    results['corr'], results['theta'] = 0, None
    results['corr_list'], results['mae_list'], results['rmse_list'] = [], [], []

    for theta in range(11):

        result_theta = smap_forecasting(X_train, y_train, X_test, y_test, theta)
        # TODO: remove X_test and y_test in smap_forecasting and move to this function

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


def edm(ts, lag, max_E, plotting=True, CV="LB"):

    results_simplex = simplex_projection(ts, lag, max_E, plotting=False, CV=CV)

    if plotting:
        plot_results(results_simplex, "simplex")

    # Embed time series and create training and test sets
    lib = embed_time_series(ts, lag=lag, E=results_simplex['E'])

    # Split in training and test set
    if CV == "RBCV":
        X_trains, y_trains, X_tests, y_tests = split_rolling_base(lib)
    else:
        X_train, y_train, X_test, y_test = split_train_test(lib, 70)
        X_trains, y_trains, X_tests, y_tests = [X_train], [y_train], [X_test], [y_test]

    results_smap = smap(X_trains, y_trains, X_tests, y_tests)

    if plotting:
        plot_results(results_smap, "s-map")

        print("Optimal theta found by S-map is " + str(results_smap['theta'])
            + r"$\rho$" + " = " + str(results_smap['corr']) + " )")

    results_smap['E'] = results_simplex['E']

    return results_smap


if __name__ == "__main__":

    print('hoi')

    #TODO: pearson correlation coefficient voor als de test set maar 1 punt bevat
