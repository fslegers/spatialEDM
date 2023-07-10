import math
import matplotlib.cm
import numpy as np
from pyEDM import *
from create_dummy_time_series import *
from preprocessing import *
from scipy.stats import pearsonr

def create_hankel_matrix(time_series, lag = 1, E = 2):
    """
    Returns the first E+1 rows of the Hankel-matrix of a time series. Each consecutive row contains
    the time series shifted backwards lag time steps.
    """
    Hankel_matrix = []

    for i in range(E + 1):
        if i == 0:
            # Add original time series
            delayed_time_series = time_series[(E - i) * lag:]
        else:
            # Add time series that is shifted i times
            delayed_time_series = time_series[(E - i) * lag:-i * lag]
        Hankel_matrix.append(delayed_time_series)

    # turn list into np.array
    Hankel_matrix = np.stack(Hankel_matrix, axis=0)

    return Hankel_matrix

def create_distance_matrix(hankel_matrix):
    N = hankel_matrix.shape[1]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            dist = np.linalg.norm((hankel_matrix[:, i] - hankel_matrix[:, j]))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix

def plot_performance_simplex(cor_list, mae_list, rmse_list):
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

def plot_results_simplex(time_series, targets, nearest_neighbors, predicted, lag, E):
    #TODO: something with weights
    time_series = create_hankel_matrix(time_series, lag, E)[0,:]
    obs_times = np.arange(0, len(time_series), 1)

    for i in range(len(targets)):
        target = targets[i]
        plt.plot(obs_times, time_series, color='black', lw=0.5)
        plt.scatter(obs_times, time_series, 5, color='black', marker='o')

        # Decide on xlim
        min_x = max(1, min(target, min(nearest_neighbors[i])) - E*lag - 1)
        max_x = min(len(time_series), max(target, max(nearest_neighbors[i])) + lag*E + 1)

        width = max_x - min_x
        if width <= 50:
            min_x = max(1, min_x - int((50 - width)/2.0)) - 1
            max_x = min(len(time_series) + 1, max_x + int((50 - width)/2.0)) + 1

        else:
            min_x = max(1, min_x - 10)
            max_x = min(len(time_series) + 1, max_x + 10)

        plt.xlim((min_x, max_x))

        # Make shaded background for target history
        if lag <= 3:
            plt.axvspan(target - E*lag + 1.1, target + 1, facecolor='m',alpha=0.2)
            for j in np.arange(target - E * lag + 1, target - lag + 1):
                plt.scatter(j, time_series[j], color = 'm', zorder=2)

        # Highlight nearest neighbors
        for neighbor in nearest_neighbors[i]:
            plt.plot([neighbor, neighbor + 1], [time_series[neighbor], time_series[neighbor + 1]],
                     linestyle='--', color='blue', lw = 2)
            plt.scatter(neighbor, time_series[neighbor], 5, color='blue', marker='o', zorder=2)
            plt.scatter(neighbor + 1, time_series[neighbor + 1], 30, color='blue', marker='D', zorder=2)

            if lag <= 3:
                plt.axvspan(neighbor - E*lag + 1.1, neighbor + 1 - 0.1, facecolor='c',alpha=0.1)

                # Highlight embedding vector
                for j in range(1, E + 1):
                    plt.scatter(neighbor - j * lag, time_series[neighbor - j * lag], 5,
                                color='blue', marker='o', zorder=2)

        # Highlight target point
        plt.plot([target, target + 1], [time_series[target], predicted[i]],
                linestyle='--', color='tab:purple', lw = 2)
        plt.scatter(target, time_series[target], 5, color='tab:purple', marker='o', zorder=2)
        plt.scatter(target + 1, time_series[target+1], 30, color='tab:purple', marker='D', zorder=2)
        plt.scatter(target + 1, predicted[i], 75, color='magenta', marker='*', zorder=2)

        # Highlight embedding vector
        if(lag <= 3):
            for j in range(1, E + 1):
                plt.scatter(target - j * lag, time_series[target - j * lag], 5, color='m', marker='o', zorder=2)

        plt.title(str(E+1) + "NN-forecast\nLag = " + str(lag) + ", E = " + str(E))
        plt.show()

    return 0

def plot_results_smap(time_series, targets, weights, predicted, lag, E):

    time_series = create_hankel_matrix(time_series, lag, E)[0,:]
    obs_times = np.arange(0, len(time_series), 1)

    #indices to step through colormap
    cmap = matplotlib.cm.get_cmap('Blues')

    for i in range(len(targets)):
        target = targets[i]
        plt.plot(obs_times, time_series, color='black', lw=0.5)
        plt.scatter(obs_times, time_series, 5, color='black', marker='o')

        # Highlight nearest neighbors
        for neighbor in range(len(time_series) - 1):
            color = cmap(0.05 + 0.95*(weights[i][neighbor] - min(weights[i]))/(max(weights[i]) - min(weights[i])))
            plt.axvspan(neighbor + 0.5, neighbor + 1.5, facecolor=color, alpha = 0.75)

        # Highlight target point
        plt.plot([target, target + 1], [time_series[target], predicted[i]],
                linestyle='--', color='tab:purple', lw = 2)
        plt.scatter(target, time_series[target], 5, color='tab:purple', marker='o', zorder=2)
        plt.scatter(target + 1, time_series[target+1], 30, color='tab:purple', marker='D', zorder=2)
        plt.scatter(target + 1, predicted[i], 75, color='magenta', marker='*', zorder=2)

        plt.title("S-Map forecast\nLag = " + str(lag) + ", E = " + str(E))
        plt.show()

    return 0

#TODO: dont consider any point that has the target in its embedding vector?
#TODO: p-step ahead prediction
def my_simplex_projection(time_series, lag = 1, max_E = 10, method = "standard"):
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
    for dim in range(1, max_E + 1):

        if method == "standard":
            # time_series should contain a single time series
            if type(time_series[0]) == list:
                print("More than one time series has been given to standard simplex projection. Proceeding with Fleurs version.")
                method = "fleur"
            else:
                hankel_matrix = create_hankel_matrix(time_series, lag, dim)
                dist_matrix = create_distance_matrix(hankel_matrix)

        if method == "fleur":
            targets = list()
            offset = 1
            hankel_matrix = np.array([]).reshape(dim+1, 0)
            for i in range(len(time_series)):
                hankel_matrix_i = create_hankel_matrix(time_series[i], lag, dim)
                N_i = hankel_matrix_i.shape[1]
                targets = targets + list(range(offset, offset + N_i - 1))
                offset += N_i
                hankel_matrix = np.hstack((hankel_matrix, hankel_matrix_i))

            dist_matrix = create_distance_matrix(hankel_matrix)
            # set distances to last values of a time series to infinity
            offset = 0
            for i in range(len(time_series)):
                index = offset + len(time_series[i]) - dim*lag - 1
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
            targets = range(N-1)

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
            nearest_neighbors = np.arange(N)[nearest_neighbors[0:dim+1]]
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
                    weight = np.exp(-dist_to_target[neighbor]/min_distance)
                    weighted_average += next_val * weight
                    total_weight += weight
                    weights.append(weight)

            weighted_average = weighted_average/total_weight

            predictions.append(weighted_average)

            # Save weights and KNNs for plotting if this dim is the optimal dim
            if target in [3, int((N-2)/2), N-3]:
                weights_per_dim.append(weights)
                KNNs_per_dim.append(nearest_neighbors)
                targets_per_dim.append(target)
                predicted_per_dim.append(weighted_average)

            #TODO: in book, they have a minimum weight of 0.000001 (why?)

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
            optimal_E = dim
            optimal_targets = targets

            weights_for_plotting = weights_per_dim
            KNNs_for_plotting = KNNs_per_dim
            targets_for_plotting = targets_per_dim
            predicted_for_plotting = predicted_per_dim

    plot_performance_simplex(cor_list, mae_list, rmse_list)

    # Plot predicted values against actual values for optimal E
    if method == 'standard':
        hankel_matrix = create_hankel_matrix(time_series, lag, E=optimal_E)
        observations = hankel_matrix[0, 1:]

    elif method == 'fleur':
        hankel_matrix = np.array([]).reshape(optimal_E + 1, 0)
        for i in range(len(time_series)):
            hankel_matrix_i = create_hankel_matrix(time_series[i], lag, E=optimal_E)
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
        plot_results_simplex(time_series,
                            targets_for_plotting,
                            KNNs_for_plotting,
                            predicted_for_plotting,
                            lag, optimal_E)

    return optimal_E

#TODO: dont consider any point that has the target in its embedding vector
#TODO: add singular value decomposition?
#TODO: p-step ahead predictions
def my_S_map(time_series, lag = 1, E = 1, method = "standard"):

    # Create Hankel matrix and Distance Matrix
    if method == "standard":
        # time_series should contain a single time series
        if type(time_series[0]) == list or type(time_series[0]) == np.ndarray:
            print(
                "More than one time series has been given to standard simplex projection. Proceeding with Fleurs version.")
            method = "fleur"
        else:
            hankel_matrix = create_hankel_matrix(time_series, lag, E)
            dist_matrix = create_distance_matrix(hankel_matrix)
            targets = range(hankel_matrix.shape[1] - 1)
    if method == "fleur":
        targets = list()
        offset = 1
        hankel_matrix = np.array([]).reshape(E + 1, 0)
        for i in range(len(time_series)):
            hankel_matrix_i = create_hankel_matrix(time_series[i], lag, E)
            N_i = hankel_matrix_i.shape[1]
            targets = targets + list(range(offset, offset + N_i - 1))
            offset += N_i
            hankel_matrix = np.hstack((hankel_matrix, hankel_matrix_i))

        dist_matrix = create_distance_matrix(hankel_matrix)
        # set distances to last values of a time series to infinity
        offset = 0
        for i in range(len(time_series)):
            index = offset + len(time_series[i]) - E * lag - 1
            offset = index + 1
            dist_matrix[:, index] = np.inf

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

    for theta in range(11):

        targets_per_theta = []
        weights_per_theta = []
        predictions_per_theta = []
        predictions = []

        # Make a one-step-ahead prediction for all points in state space
        # except the last observed point
        for target in range(len(targets)):
            distances = dist_matrix[target, :]
            distances[target] = np.inf
            distances_no_inf = distances[distances < np.inf]


            if d_m == 0:
                #TODO: work out this scenario
                print('Distance to all points is zero.')
                return 0

            # Calculate weights
            mean_dist = mean(distances_no_inf)
            weights = np.exp(-theta * distances_no_inf / mean_dist)
            next_vals = Hankel_matrix[0, 1:]
            next_vals = next_vals[distances < np.inf]

            B = np.multiply(weights, next_vals)
            A = np.empty((len(weights), 0))

            # Fill A
            for i in range(1, E+1):
                prev_value = Hankel_matrix[i, ]
                prev_value = prev_value[distances < np.inf]
                new_column = np.multiply(weights, prev_value)
                A = np.hstack((A, new_column))

            # Calculate coefficients C using the pseudo-inverse of A (via SVD)
            coeffs = np.linalg.pinv(A) * B

            # Make prediction
            next_val = Hankel_matrix[:, target] * coeffs

            #weights[dist_matrix[target, :-1] == np.inf] = 0
            #weights[target] = 0
            #next_val = np.dot(weights, np.transpose(hankel_matrix[0, 1:])) / sum(weights)
            predictions.append(next_val)

            if target in [targets[2], targets[int(len(targets)/2)], targets[len(targets) - 2]]:
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
        observations.append(hankel_matrix[0,target])

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
        plot_results_smap(time_series, targets_for_plotting, weights_for_plotting, predictions_for_plotting, lag, E)

    return optimal_theta

if __name__ == "__main__":
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
    # time_series_a = time_series[1:250]
    # time_series_b = time_series[10:261]
    # time_series_c = time_series[20:272]
    # time_series = [time_series_a, time_series_b, time_series_c]

    # Plot time series
    #plot_time_series(time_series)

    #plot_autocorrelation(time_series)
    #plot_partial_autocorrelation(time_series)
    #plot_recurrence(time_series[1:100], delay=8)
    #make_lag_scatterplot(time_series, lag=8)

    time_series = np.arange(1,100)
    time_series = time_series/100
    time_series = np.sin(time_series)

    #optimal_E = my_simplex_projection(time_series, lag=8, max_E=10, method='fleur')
    my_S_map(time_series, lag=1, E=3)
