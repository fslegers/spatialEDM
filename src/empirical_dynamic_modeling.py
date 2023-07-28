import math

import itertools

import matplotlib.cm
from pyEDM import *
from scipy.stats import pearsonr

from preprocessing import *
from src.create_dummy_time_series import simulate_lorenz
from src.time_series_plots import plot_time_series


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


def make_libraries(ts, lag, E, test_interval):
    """
    Function that takes a single or multiple time series, concatenates them if necessary,
    creates tuples of time-delay embedding vectors and one-step-ahead predictees and returns
    these in a training and test set.
    :param ts: a time series or a list of time series
    :param test_interval: a tuple [x, y] marking the interval for test instances
    :return: a training and test set of tuples [[x_t-1, ..., x_t-E], x_t]
    """

    lib = []

    # make sure time_series is a list of np_arrays
    if isinstance(ts, list):
        if isinstance(ts[0], float) or isinstance(ts[0], int):
            ts = np.array(ts)
            ts = [ts]
        if isinstance(ts[0], list):
            for i in range(len(ts)):
                ts[i] = np.array(ts[i])

    for series in range(len(ts)):
        hankel_matrix = create_hankel_matrix(ts[series], lag, E)

        # For each column, create a tuple (time-delay vector, prediction)
        for col in range(hankel_matrix.shape[1]):
            tuple = [hankel_matrix[1:, col], hankel_matrix[0, col]]
            lib.append(tuple)

    if test_interval[0] < 0 or test_interval[1] < test_interval[0] or test_interval[1] > len(lib):
        print("Given test interval is invalid. Splitting the library in two equal halves.")
        test_interval[0] = int(len(lib) / 2)
        test_interval[1] = len(lib)

    # split into training and test set
    test_set = lib[test_interval[0]:test_interval[1]]
    training_set = lib[:test_interval[0]] + lib[test_interval[1]:]

    return(training_set, test_set)

def create_distance_matrix(hankel_matrix):
    """
    Returns a matrix of distances between points (columns of the Hankel matrix) in state space.
    """
    N = hankel_matrix.shape[1]
    dist_matrix = np.zeros((N, N))
    for p in range(N):
        for q in range(p, N):
            dist = np.linalg.norm((hankel_matrix[:, p] - hankel_matrix[:, q]))
            dist_matrix[p, q] = dist
            dist_matrix[q, p] = dist

    return dist_matrix


def plot_performance_simplex(cor_list, mae_list, rmse_list):
    """
    Plots the correlation coefficient, mean absolute error (MAE) and
    root mean square error (RMSE) of Simplex for each dimension E.
    """
    # Show figure of performance plots
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Performance measures per E')
    max_E = len(cor_list) + 1

    axs[0].set_ymargin(0.1)
    axs[1].set_ymargin(0.1)
    axs[2].set_ymargin(0.1)

    axs[0].plot(range(1, len(cor_list) + 1), cor_list, color='black', marker='o')
    axs[1].plot(range(1, len(mae_list) + 1), mae_list, color='black', marker='o')
    axs[2].plot(range(1, len(rmse_list) + 1), rmse_list, color='black', marker='o')

    axs[0].set_ylabel('rho')
    axs[1].set_ylabel('MAE')
    axs[2].set_ylabel('RMSE')
    axs[2].set_xlabel('E')

    major_tick = range(1, max_E + 1)
    axs[0].set_xticks(major_tick)
    axs[0].xaxis.grid(which='major')
    axs[1].xaxis.grid(True)
    axs[2].xaxis.grid(True)
    axs[0].ticklabel_format(useOffset=False)
    axs[0].yaxis.TickLabelFormat = '%.2f'

    # Highlight the point with optimal performance measure
    axs[0].plot(np.argmax(cor_list) + 1, max(cor_list), color='m', marker='D', markersize=7)
    axs[1].plot(np.argmin(mae_list) + 1, min(mae_list), color='m', marker='D', markersize=7)
    axs[2].plot(np.argmin(rmse_list) + 1, min(rmse_list), color='m', marker='D', markersize=7)

    plt.show()

    print("Highest correlation for E = :", str(np.argmax(cor_list) + 1) + " (" + str(max(cor_list)) + ")")
    print("Lowest MAE for E = :", str(np.argmin(mae_list) + 1) + " (" + str(min(mae_list)) + ")")
    print("Lowest RMSE for E = :", str(np.argmin(rmse_list) + 1) + " (" + str(min(rmse_list)) + ")")


def plot_results_simplex(ts, targets, nearest_neighbors, predicted, lag, E):
    """
    Shows which neighbors were used in the prediction of the target points "targets" by the Simplex
    method with dimension E. Plots both the predicted as the observed values of targets next value.
    """
    # TODO: something with weights
    ts = create_hankel_matrix(ts, lag, E)[0, :]
    obs_times = np.arange(0, len(ts), 1)

    for j in range(len(targets)):
        target = targets[j]
        plt.plot(obs_times, ts, color='black', lw=0.5)
        plt.scatter(obs_times, ts, 5, color='black', marker='o')

        # Decide on xlim
        min_x = max(1, min(target, min(nearest_neighbors[j])) - E * lag - 1)
        max_x = min(len(ts), max(target, max(nearest_neighbors[j])) + lag * E + 1)

        width = max_x - min_x
        if width <= 50:
            min_x = max(1, min_x - int((50 - width) / 2.0)) - 1
            max_x = min(len(ts) + 1, max_x + int((50 - width) / 2.0)) + 1

        else:
            min_x = max(1, min_x - 10)
            max_x = min(len(ts) + 1, max_x + 10)

        plt.xlim((min_x, max_x))

        # Make shaded background for target history
        if lag <= 3:
            plt.axvspan(target - E * lag + 1.1, target + 1, facecolor='m', alpha=0.2)
            for j in np.arange(target - E * lag + 1, target - lag + 1):
                plt.scatter(j, ts[j], color='m', zorder=2)

        # Highlight nearest neighbors
        for neighbor in nearest_neighbors[j]:
            plt.plot([neighbor, neighbor + 1], [ts[neighbor], ts[neighbor + 1]],
                     linestyle='--', color='blue', lw=2)
            plt.scatter(neighbor, ts[neighbor], 5, color='blue', marker='o', zorder=2)
            plt.scatter(neighbor + 1, ts[neighbor + 1], 30, color='blue', marker='D', zorder=2)

            if lag <= 3:
                plt.axvspan(neighbor - E * lag + 1.1, neighbor + 1 - 0.1, facecolor='c', alpha=0.1)

                # Highlight embedding vector
                for j in range(1, E + 1):
                    plt.scatter(neighbor - j * lag, ts[neighbor - j * lag], 5,
                                color='blue', marker='o', zorder=2)

        # Highlight target point
        plt.plot([target, target + 1], [ts[target], predicted[j]],
                 linestyle='--', color='tab:purple', lw=2)
        plt.scatter(target, ts[target], 5, color='tab:purple', marker='o', zorder=2)
        plt.scatter(target + 1, ts[target + 1], 30, color='tab:purple', marker='D', zorder=2)
        plt.scatter(target + 1, predicted[j], 75, color='magenta', marker='*', zorder=2)

        # Highlight embedding vector
        if lag <= 3:
            for q in range(1, E + 1):
                plt.scatter(target - q * lag, ts[target - q * lag], 5, color='m', marker='o', zorder=2)

        plt.title(str(E + 1) + "NN-forecast\nLag = " + str(lag) + ", E = " + str(E))
        plt.show()

    return 0


def plot_results_smap(ts, targets, weights, predicted, distances, lag, E):
    ts = create_hankel_matrix(ts, lag, E)[0, :]
    obs_times = np.arange(0, len(ts), 1)

    # indices to step through colormap
    cmap = matplotlib.cm.get_cmap('Blues')

    for i in range(len(targets)):
        target = targets[i]
        plt.plot(obs_times, ts, color='black', lw=0.5)
        plt.scatter(obs_times, ts, 5, color='black', marker='o')

        # Highlight nearest neighbors
        neighborhood = list(itertools.compress(range(len(ts) - 1), distances[target, :-1]))
        for neighbor in range(len(neighborhood)):
            color = cmap(0.05 + 0.95 * (weights[i][neighbor] - min(weights[i])) / (max(weights[i]) - min(weights[i])))
            plt.axvspan(neighborhood[neighbor] - 0.5, neighborhood[neighbor] + 0.5, facecolor=color, alpha=0.75)

        # Highlight target point
        plt.plot([target, target + 1], [ts[target], predicted[i]],
                 linestyle='--', color='tab:purple', lw=2)
        plt.scatter(target, ts[target], 5, color='tab:purple', marker='o', zorder=2)
        plt.scatter(target + 1, ts[target + 1], 30, color='tab:purple', marker='D', zorder=2)
        plt.scatter(target + 1, predicted[i], 75, color='magenta', marker='*', zorder=2)

        plt.title("S-Map forecast\nLag = " + str(lag) + ", E = " + str(E))
        plt.show()

    return 0


# TODO: dont consider any point that has the target in its embedding vector?
# TODO: p-step ahead prediction
def simplex_projection(ts, lag=1, max_E=10, method="standard"):
    """
    Simplex projecting with leave-one-out cross validation. Finds an optimal embedding dimension E that maximizes the
    correlation coefficient between predicted and observed values.
    """

    # Things to keep track of for plotting
    cor_list = []
    mae_list = []
    rmse_list = []
    KNNs_for_plotting = []
    weights_for_plotting = []
    targets_for_plotting = []
    predicted_for_plotting = []

    # Things to keep track of for finding optimal E
    optimal_cor = 0
    optimal_predictions = []
    optimal_targets = []

    # For each dimension E
    max_E = int(max_E)
    for dim in range(1, max_E + 1):

        if method == "standard":
            # time_series should contain a single time series
            if type(ts[0]) == list:
                print("More than one time series has been given to standard simplex projection. "
                      "Proceeding with Hsieh's version.")
                method = "dewdrop"
            else:
                hankel_matrix = create_hankel_matrix(ts, lag, dim)
                dist_matrix = create_distance_matrix(hankel_matrix)

        if method == "dewdrop":
            targets = list()
            offset = 1
            hankel_matrix = np.array([]).reshape(dim + 1, 0)
            for i in range(len(ts)):
                hankel_matrix_i = create_hankel_matrix(ts[i], lag, dim)
                N_i = hankel_matrix_i.shape[1]
                targets = targets + list(range(offset, offset + N_i - 1))
                offset += N_i
                hankel_matrix = np.hstack((hankel_matrix, hankel_matrix_i))

            dist_matrix = create_distance_matrix(hankel_matrix)
            # set distances to last values of a time series to infinity
            offset = 0
            for i in range(len(ts)):
                index = offset + len(ts[i]) - dim * lag - 1
                offset = index + 1
                dist_matrix[:, index] = np.inf

        predictions = []
        N = hankel_matrix.shape[1]

        KNNs_per_dim = []
        weights_per_dim = []
        targets_per_dim = []
        predicted_per_dim = []

        # for all target points, get dim+1 nearest neighbors and make one-step-ahead prediction (weighted average)
        if method == 'standard':
            targets = range(N - 1)

        for target in targets:

            # Exclude target point and last point
            # by temporarily setting their value to infinity
            dist_to_target = dist_matrix[target, :]
            if target == 0:
                dist_to_target[0] = np.inf
            else:
                dist_to_target[target] = np.inf
            if method == 'standard':
                dist_to_target[N - 1] = np.inf

            # Select E + 1 nearest neigbors
            nearest_neighbors = np.argpartition(dist_to_target, (0, dim + 2))
            nearest_neighbors = np.arange(N)[nearest_neighbors[0:dim + 1]]
            min_distance = dist_to_target[nearest_neighbors[0]]

            weighted_average = 0
            total_weight = 0
            weights = []

            if min_distance == 0:
                for neighbor in nearest_neighbors:
                    if dist_to_target[neighbor] == 0:
                        weight = 1
                    else:
                        weight = 0.000001
                    next_val = hankel_matrix[0, neighbor + 1]
                    weighted_average += next_val * weight
                    total_weight += weight
                    weights.append(weight)

            else:
                for neighbor in nearest_neighbors:
                    # Add next value to weighted average
                    next_val = hankel_matrix[0, neighbor + 1]
                    weight = np.exp(-dist_to_target[neighbor] / min_distance)
                    weighted_average += next_val * weight
                    total_weight += weight
                    weights.append(weight)

            weighted_average = weighted_average / total_weight

            predictions.append(weighted_average)

            # Save weights and KNNs for plotting if this dim is the optimal dim
            if target in [3, int((N - 2) / 2), N - 3]:
                weights_per_dim.append(weights)
                KNNs_per_dim.append(nearest_neighbors)
                targets_per_dim.append(target)
                predicted_per_dim.append(weighted_average)

            # TODO: in book, they have a minimum weight of 0.000001 (why?)

        # Pearson Correlation Coefficient
        if method == 'standard':
            observations = hankel_matrix[0, 1:]
        elif method == 'fleur':
            observations = []
            for target in targets:
                observations.append(hankel_matrix[0, target])

        cor = pearsonr(observations, predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(observations, predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(observations, predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

        if cor >= optimal_cor:
            optimal_cor = cor
            optimal_predictions = predictions
            optimal_dim = dim
            optimal_targets = targets

            weights_for_plotting = weights_per_dim
            KNNs_for_plotting = KNNs_per_dim
            targets_for_plotting = targets_per_dim
            predicted_for_plotting = predicted_per_dim

    plot_performance_simplex(cor_list, mae_list, rmse_list)

    # Plot predicted values against actual values for optimal E
    if method == 'standard':
        hankel_matrix = create_hankel_matrix(ts, lag, E=optimal_dim)
        observations = hankel_matrix[0, 1:]

    elif method == 'fleur':
        hankel_matrix = np.array([]).reshape(optimal_dim + 1, 0)
        for i in range(len(ts)):
            hankel_matrix_i = create_hankel_matrix(ts[i], lag, E=optimal_dim)
            print(np.shape(hankel_matrix_i)[1])
            hankel_matrix = np.hstack((hankel_matrix, hankel_matrix_i))

        observations = []
        for target in optimal_targets:
            observations.append(hankel_matrix[0, target])

    xmin = min(min(optimal_predictions), min(observations))
    xmax = max(max(optimal_predictions), max(observations))

    xmin = xmin - 0.1 * np.abs(xmin)
    xmax = xmax + 0.1 * np.abs(xmax)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))

    plt.plot([xmin, xmax], [xmin, xmax], color='black')
    plt.scatter(observations, optimal_predictions, color='black')

    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Simplex results for E = " + str(np.argmax(cor_list) + 1))

    plt.show()

    if method == 'standard':
        plot_results_simplex(ts,
                             targets_for_plotting,
                             KNNs_for_plotting,
                             predicted_for_plotting,
                             lag, optimal_dim)

    return optimal_dim


# TODO: dont consider any point that has the target in its embedding vector
# TODO: add singular value decomposition?
# TODO: p-step ahead predictions
def smap(ts, lag=1, E=1):
    hankel_matrix = create_hankel_matrix(ts, lag, E)
    dist_matrix = create_distance_matrix(hankel_matrix)
    targets = range(hankel_matrix.shape[1] - 1)

    if method == "fleur":
        targets = list()
        offset = 1
        hankel_matrix = np.array([]).reshape(E + 1, 0)
        for i in range(len(ts)):
            hankel_matrix_i = create_hankel_matrix(ts[i], lag, E)
            N_i = hankel_matrix_i.shape[1]
            targets = targets + list(range(offset, offset + N_i - 1))
            offset += N_i
            hankel_matrix = np.hstack((hankel_matrix, hankel_matrix_i))

        dist_matrix = create_distance_matrix(hankel_matrix)
        # set distances to last values of a time series to infinity
        offset = 0
        for i in range(len(ts)):
            index = offset + len(ts[i]) - E * lag - 1
            offset = index + 1
            dist_matrix[:, index] = np.inf

    # set distances of diagonal to infinity
    ind = np.diag_indices_from(dist_matrix)
    dist_matrix[ind] = np.inf

    N = hankel_matrix.shape[1]
    cor_list = []
    mae_list = []
    rmse_list = []

    optimal_cor = 0
    optimal_theta = 0
    optimal_predictions = []

    targets_for_plotting = []
    weights_for_plotting = []
    predictions_for_plotting = []
    neighbors_for_plotting = []

    for theta in range(11):

        targets_per_theta = []
        weights_per_theta = []
        predictions_per_theta = []
        predictions = []

        # Make a one-step-ahead prediction for all points in state space
        # except the last observed point
        for target in range(len(targets)):
            distances = dist_matrix[target, :-1]  # Exclude last point
            # distances[target] = np.inf
            distances_no_inf = distances[distances < np.inf]

            # if d_m == 0:
            #    #TODO: work out this scenario
            #    print('Distance to all points is zero.')
            #    return 0

            # Calculate weights
            mean_dist = mean(distances_no_inf)
            weights = np.exp(-theta * distances_no_inf / mean_dist)
            next_vals = hankel_matrix[0, 1:]
            next_vals = list(itertools.compress(next_vals, distances < np.inf))

            B = np.multiply(weights, next_vals)
            A = np.empty((len(weights),))

            # Fill A
            for i in range(1, E + 1):
                prev_value = hankel_matrix[i, :-1]
                prev_value = list(itertools.compress(prev_value, distances < np.inf))
                new_column = np.multiply(weights, prev_value)
                A = np.vstack((A, new_column))

            # Calculate coefficients C using the pseudo-inverse of A (via SVD)
            A = np.transpose(A)
            coeffs = np.matmul(np.linalg.pinv(A), B)

            # Make prediction
            next_val = np.matmul(hankel_matrix[:, target], coeffs)

            # weights[dist_matrix[target, :-1] == np.inf] = 0
            # weights[target] = 0
            # next_val = np.dot(weights, np.transpose(hankel_matrix[0, 1:])) / sum(weights)
            predictions.append(next_val)

            if target in [targets[2], targets[int(len(targets) / 2)], targets[len(targets) - 2]]:
                targets_per_theta.append(target)
                weights_per_theta.append(weights)
                predictions_per_theta.append(predictions[target])

        # Pearson Correlation Coefficient
        if method == 'standard':
            observations = hankel_matrix[0, 1:]
        elif method == 'fleur':
            observations = []
            for target in targets:
                observations.append(hankel_matrix[0, target])

        # Pearson Correlation Coefficient
        cor = pearsonr(observations, predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(observations, predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(observations, predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

        # Update optimal predictions
        if cor >= optimal_cor:
            optimal_theta = theta
            optimal_cor = cor
            optimal_predictions = predictions

            targets_for_plotting = targets_per_theta
            weights_for_plotting = weights_per_theta
            predictions_for_plotting = predictions_per_theta
            neighbors_for_plotting = dist_matrix < np.inf

    # Show figure of performance plots
    plt.figure(0)
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Performance measures per Theta')

    axs[0].set_ymargin(0.1)
    axs[1].set_ymargin(0.1)
    axs[2].set_ymargin(0.1)

    axs[0].plot(range(0, 11), cor_list, color='black', marker='o')
    axs[1].plot(range(0, 11), mae_list, color='black', marker='o')
    axs[2].plot(range(0, 11), rmse_list, color='black', marker='o')

    axs[0].set_ylabel('rho')
    axs[1].set_ylabel('MAE')
    axs[2].set_ylabel('RMSE')
    axs[2].set_xlabel('theta')

    major_tick = range(0, 11)
    axs[0].set_xticks(major_tick)
    axs[0].xaxis.grid(which='major')
    axs[0].xaxis.grid(True)
    axs[1].xaxis.grid(True)
    axs[2].xaxis.grid(True)

    axs[0].plot(np.argmax(cor_list), max(cor_list), color='m', marker='D', markersize=7)
    axs[1].plot(np.argmin(mae_list), min(mae_list), color='m', marker='D', markersize=7)
    axs[2].plot(np.argmin(rmse_list), min(rmse_list), color='m', marker='D', markersize=7)

    plt.show()

    # Plot predicted values against actual values for optimal theta
    plt.figure(1)
    observations = []

    for target in targets:
        observations.append(hankel_matrix[0, target])

    xmin = min(min(optimal_predictions), min(observations))
    xmax = max(max(optimal_predictions), max(observations))

    xmin = xmin - 0.1 * np.abs(xmin)
    xmax = xmax + 0.1 * np.abs(xmax)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))

    plt.plot([xmin, xmax], [xmin, xmax], color='black')
    plt.scatter(observations, optimal_predictions, color='black')

    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Scatter plot for E = " + str(E) + r" and $\theta$ = " + str(optimal_theta))

    plt.show()

    if method == 'standard':
        plot_results_smap(ts,
                          targets_for_plotting,
                          weights_for_plotting,
                          predictions_for_plotting,
                          neighbors_for_plotting,
                          lag,
                          E)

    return optimal_theta


if __name__ == "__main__":
    a = [1,2,3,4,5,6,7,8,9,10]
    # train, test = make_libraries(a, 1, 3, [2,4])

    b = [1,2,3,4,5,6,7,8,9,10]
    train, test = make_libraries([a,b], 1, 3, [-6,4])

    a = np.array(a)
    b = np.array(b)
    train, test = make_libraries([a, b], 1, 3, [2, 90])

    # # Sample lorenz trajectory
    # lorenz_trajectory = simulate_lorenz(t_max=2500, noise=0.01)
    # lorenz_x = lorenz_trajectory[850:, 0]
    #
    # new_lorenz_x = []
    # for i in range(len(lorenz_x)):
    #     if i % 10 == 0:
    #         new_lorenz_x.append(lorenz_x[i])
    #
    # lorenz_x = new_lorenz_x
    #
    # # Differentiate and standardize
    # time_series = np.diff(lorenz_x)
    # time_series = standardize_time_series(time_series)
    # time_series = time_series[:, 0]
    #
    # time_series_a = np.array(time_series[1:250])
    # time_series_b = np.array(time_series[10:261])
    # time_series_c = np.array(time_series[20:272])
    # time_series = [time_series_a, time_series_b, time_series_c]
    #
    # make_libraries(time_series, 1, 3, [10,30])
    #
    # # Plot time series
    # # plot_time_series(time_series)
    #
    # # plot_autocorrelation(time_series)
    # # plot_partial_autocorrelation(time_series)
    # # plot_recurrence(time_series[1:100], delay=8)
    # # make_lag_scatterplot(time_series, lag=8)
    #
    # # time_series = np.arange(1,100)
    # # time_series = time_series/100
    # # time_series = np.sin(time_series)
    #
    # #optimal_E = simplex_projection(time_series, lag=8, max_E=10, method="dewdrop")
    # #smap(time_series, lag=8, E=optimal_E, method="dewdrop")

    #TODO:
    #time series should be np.arrays, not lists
    #somewhere, this should be checked