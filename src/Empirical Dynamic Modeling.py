import math

import numpy as np
from pyEDM import *
import pandas as pd
from create_dummy_time_series import *
from preprocessing import *
from sklearn import preprocessing
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

    return(Hankel_matrix)

def create_distance_matrix(hankel_matrix):
    N = hankel_matrix.shape[1]
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            dist = np.linalg.norm((hankel_matrix[:, i] - hankel_matrix[:, j]))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return(dist_matrix)

def plot_result_in_time_series(time_series, targets, nearest_neighbors, weights, predicted, lag = 1, E = 1):

    obs_times = np.arange(1, 1 + len(time_series), 1)

    for i in range(len(targets)):

        plt.plot(obs_times, time_series, color='black', lw=1)
        plt.scatter(obs_times, time_series, 5, color='black', marker='o')

        # Decide on xlim
        min_x = min(targets[i], min(nearest_neighbors[i])) - 1
        max_x = max(targets[i], max(nearest_neighbors[i])) + E*lag + 1

        width = max_x - min_x
        if width <= 50:
            min_x = max(1, min_x - int((50 - width)/2.0)) - 1
            max_x = min(len(time_series) + 1, max_x + int((50 - width)/2.0)) + 1

        else:
            min_x = max(1, min_x - 10)
            max_x = min(len(time_series) + 1, max_x + 10)

        plt.xlim((min_x, max_x))

        # Make shaded background for target history
        #plt.axvspan(target - E*lag + 1.1, target + 1, facecolor='m',alpha=0.2)
        #for j in np.arange(target - E * lag + 1, target - lag + 1):
            #plt.scatter(j, time_series[j], color = 'm', zorder=2)

        # Highlight nearest neighbors
        for neighbor in nearest_neighbors[i]:
            neighbor = int(neighbor + E*lag + 1)
            plt.plot([neighbor + 1, neighbor + 1 + lag], [time_series[neighbor], time_series[neighbor + lag]],
                     linestyle='--', color='tab:blue')
            plt.scatter(neighbor + 1, time_series[neighbor], 40, color='blue', marker='s', zorder=2)
            plt.scatter(neighbor + 1 + lag, time_series[neighbor + lag], 35, color='blue', marker='D', zorder=2)
            plt.axvspan(neighbor - E*lag + 1.1, neighbor + 1 - 0.1, facecolor='c',alpha=0.1)

            # Highlight embedding vector
            for j in range(1, E + 1):
                plt.scatter(neighbor - j * lag + 1, time_series[neighbor - j * lag], 5, color='blue', marker='o', zorder=2)

        # Highlight target point
        target = int(targets[i] + E * lag)
        plt.plot([target + 1, target + 1 + lag], [time_series[target], predicted[i]],
                linestyle='--', color='tab:purple')
        plt.scatter(target + 1, time_series[target], 40, color='m', marker='s', zorder=2)
        plt.scatter(target + 1 + lag, time_series[target + lag], 35, color='m', marker='D', zorder=2)
        plt.scatter(target + 1 + lag, predicted[i], 60, color='m', marker='*', zorder=2)

        # Highlight embedding vector
        for j in range(1, E + 1):
            plt.scatter(target - j * lag + 1, time_series[target - j * lag], 5, color='m', marker='o', zorder=2)

        plt.title(str(E+1) + "NN-forecast\nLag = " + str(lag) + ", E = " + str(E))
        plt.show()

    return 0

def simplex_projection(time_series, lag = -1, max_E = 10):
    """
    Finds the optimal value for the embedding dimension E by one-step-ahead predictions
    using E+1 Nearest Neighbors.
    :return: E, optimal dimension
    """
    #Check if time_series is standardized
    if np.abs(mean(time_series)) > 2e-5 or np.abs(std(time_series)) - 1 > 2e-5:
        print("standardizing time series...")
        time_series = preprocessing.scale(time_series)

    # If no observation times are given, add them to time_series
    if len(np.shape(time_series)) == 1:
        obs_times = np.arange(1, np.shape(time_series)[0] + 1, 1)
        time_series = np.column_stack((obs_times, time_series))

    # Turn time_series into pandas dataframe
    df = pd.DataFrame(time_series, columns = ["t", "x"])

    # Divide time series in training (60%), and test set (40%)
    length = len(df)
    training_set = "1 " + str(floor(0.6 * length))
    test_set = str(floor(0.6 * length) + 1) + " " + str(length)

    # Plot and return prediction skill rho for each embedding via Simplex
    rho_per_E = EmbedDimension(dataFrame = df, maxE = max_E, tau = -np.abs(lag),
                               lib = training_set, pred = test_set, columns = "x")

    # Find the optimal E
    optimal_param = rho_per_E.loc[rho_per_E['rho'].idxmax()]
    print("Optimal embedding dimension E: ", str(optimal_param["E"]),
          " ( rho = ", str(optimal_param["rho"]), ").")

    return(optimal_param["E"])

def S_map(time_series, obs_time, lag=-1, E=10):
    """
    Evaluates Smap prediction skill for different values of localization parameter theta)
    :param time_series:
    :param lag:
    :param E:
    :return:
    """
    # TODO
    # Add check if time_series is standardized

    # If no observation times are given, add them to time_series
    if len(np.shape(time_series)) == 1:
        obs_times = np.arange(1, np.shape(time_series)[0] + 1, 1)
        time_series = np.column_stack((obs_times, time_series))

    # Turn time_series into pandas dataframe (column vector)
    df = pd.DataFrame(time_series, columns=["t", "x"])

    # Divide time series in training (60%), and test set (40%)
    length = len(df)
    training_set = "1 " + str(floor(0.6 * length))
    test_set = str(floor(0.6 * length) + 1) + " " + str(length)

    # Evaluate SMap prediction skill for localization parameter theta
    rho_per_theta = PredictNonlinear(dataFrame=df, E=E, tau=-np.abs(lag),
                                     lib=training_set, pred=test_set, columns="x", embedded=False)

    return (rho_per_theta)

def my_simplex_projection(time_series, lag = 1, max_E = 10, show_plots = False):
    #TODO: think about embedding dimension: number of lags or dimension of state space?
    cor_list = []
    mae_list = []
    rmse_list = []

    optimal_cor = 0
    optimal_predictions = []

    KNNs_for_plotting = []
    weights_for_plotting = []
    predicted_for_plotting = []

    # For each dimension E
    for dim in range(1, max_E + 1):
        hankel_matrix = create_hankel_matrix(time_series, lag, dim)
        dist_matrix = create_distance_matrix(hankel_matrix)

        predictions = []
        N = hankel_matrix.shape[1]

        KNNs_per_dim = []
        weights_per_dim = []
        predicted_per_dim = []

        # for all target points, get dim+1 nearest neighbors and make one-step-ahead prediction (weighted average)
        for target in range(N-1):
            # find dim + 1 nearest neighbors
            nearest_neighbors = np.argpartition(dist_matrix[target, :-1], (0, dim + 2)) # exclude last point because a next value has to be available
            nearest_neighbors = np.arange(N)[nearest_neighbors[1:dim+2]]

            min_distance = dist_matrix[target, nearest_neighbors[0]]
            weighted_average = 0
            total_weight = 0
            weights = []

            # if min_distance = 0, next_val will be average of points
            if min_distance == 0:
                i = 0
                weighted_average = 0
                while dist_matrix[target, nearest_neighbors[i]] == 0:
                    weighted_average += time_series[nearest_neighbors[i]]
                    i += 1
                weighted_average = weighted_average/(i + 1)

            else:
                for neighbor in nearest_neighbors:
                    # Add next value to weighted average
                    next_val = time_series[neighbor + 1]
                    weight = np.exp(-dist_matrix[target, neighbor]/min_distance)
                    total_weight += weight
                    weighted_average += next_val * weight
                    weights.append(weight)

                weighted_average = weighted_average/total_weight

            predictions.append(weighted_average)

            # Save weights and KNNs for plotting if this dim is the optimal dim
            if target in [0, int((N-2)/2), N-2]:
                weights_per_dim.append(weights)
                KNNs_per_dim.append(nearest_neighbors)
                predicted_per_dim.append(weighted_average)

            # plot time series with neighbors to make result more insightful
            # do this for first, last and middle point
            #if target in [0, N-1, int((N-1)/2)] and min_distance > 0:
                #plot_result_in_time_series(time_series, target, nearest_neighbors, weights, lag, dim)

            #TODO: in book, they have a minimum weight of 0.000001 (why?)

        if show_plots:
            plt.scatter(hankel_matrix[0, 1:], predictions)
            plt.plot(range(0, int(max(time_series))), range(0, int(max(time_series))))
            plt.title("E = " + str(dim))
            plt.xlabel("Observed values", fontsize = 12)
            plt.ylabel("Predicted values", fontsize = 12)
            plt.show()

        # Pearson Correlation Coefficient
        cor = pearsonr(hankel_matrix[0, 1:], predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(hankel_matrix[0, 1:], predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(hankel_matrix[0, 1:], predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

        if cor >= optimal_cor:
            optimal_cor = cor
            optimal_predictions = predictions
            optimal_E = dim

            weights_for_plotting = weights_per_dim
            KNNs_for_plotting = KNNs_per_dim
            predicted_for_plotting = predicted_per_dim

    # Show figure of performance plots
    fig, axs = plt.subplots(3, sharex = True)
    fig.suptitle('Performance measures per E')

    axs[0].set_ymargin(0.1)
    axs[1].set_ymargin(0.1)
    axs[2].set_ymargin(0.1)

    axs[0].plot(range(1,len(cor_list)+1), cor_list, color = 'black', marker = 'o')
    axs[1].plot(range(1, len(mae_list)+1), mae_list, color = 'black', marker = 'o')
    axs[2].plot(range(1, len(rmse_list)+1), rmse_list, color = 'black', marker = 'o')

    axs[0].set_ylabel('rho')
    axs[1].set_ylabel('MAE')
    axs[2].set_ylabel('RMSE')
    axs[2].set_xlabel('E')

    #axs[0].set_ylim(min(cor_list)-0.005, 1)

    major_tick = range(1, max_E + 1)
    axs[0].set_xticks(major_tick)
    axs[0].xaxis.grid(which='major')
    axs[1].xaxis.grid(True)
    axs[2].xaxis.grid(True)
    axs[0].ticklabel_format(useOffset=False)
    axs[0].yaxis.TickLabelFormat = '%.2f'

    # Highlight the point with optimal performance measure
    axs[0].plot(np.argmax(cor_list) + 1, max(cor_list), color='m', marker='D', markersize = 7)
    axs[1].plot(np.argmin(mae_list) + 1, min(mae_list), color='m', marker='D', markersize = 7)
    axs[2].plot(np.argmin(rmse_list) + 1, min(rmse_list), color='m', marker='D', markersize = 7)

    plt.show()

    print("Highest correlation for E = :", str(np.argmax(cor_list) + 1) + " (" + str(max(cor_list)) + ")")
    print("Lowest MAE for E = :", str(np.argmin(mae_list) + 1) + " (" + str(min(mae_list)) + ")")
    print("Lowest RMSE for E = :", str(np.argmin(rmse_list) + 1) + " (" + str(min(rmse_list)) + ")")

    # Plot predicted values against actual values for optimal E
    hankel_matrix = create_hankel_matrix(time_series, lag, E=optimal_E)
    xmin = min(min(optimal_predictions), min(hankel_matrix[0, 1:]))
    xmax = max(max(optimal_predictions), max(hankel_matrix[0, 1:]))

    xmin = xmin - 0.1 * np.abs(xmin)
    xmax = xmax + 0.1 * np.abs(xmax)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))

    plt.plot([xmin, xmax], [xmin, xmax], color='black')
    plt.scatter(hankel_matrix[0, 1:], optimal_predictions, color='black')

    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Simplex results for E = " + str(np.argmax(cor_list) + 1))

    plt.show()

    plot_result_in_time_series(time_series, [0, int(N-2)/2, N-2],
                               KNNs_for_plotting,
                               weights_for_plotting,
                               predicted_for_plotting,
                               lag, optimal_E)

    return optimal_E

def my_S_map(time_series, lag = 1, E = 2):
    hankel_matrix = create_hankel_matrix(time_series, lag, E)
    dist_matrix = create_distance_matrix(hankel_matrix)

    N = hankel_matrix.shape[1]
    cor_list = []
    mae_list = []
    rmse_list = []

    optimal_cor = 0
    optimal_theta = 0
    optimal_predictions = []

    for theta in range(11):

        predictions = []

        # Make a one-step-ahead prediction for all points in state space
        # except the last observed point
        for target in range(N-1):
            d_m = mean(np.concatenate((dist_matrix[target, :target], dist_matrix[target, target+1:-1])))

            if d_m == 0:
                print('Distance to all points is zero.')
                return 0

            weights = np.exp(-theta * dist_matrix[target, :-1] / d_m)
            weights[target] = 0
            next_val = np.dot(weights, np.transpose(hankel_matrix[0, 1:])) / sum(weights)
            predictions.append(next_val)

        # Pearson Correlation Coefficient
        cor = pearsonr(hankel_matrix[0, 1:], predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(hankel_matrix[0, 1:], predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(hankel_matrix[0, 1:], predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

        # Update optimal predictions
        if cor >= optimal_cor:
            optimal_theta = theta
            optimal_cor = cor
            optimal_predictions = predictions

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
    xmin = min(min(optimal_predictions), min(hankel_matrix[0, 1:]))
    xmax = max(max(optimal_predictions), max(hankel_matrix[0, 1:]))

    xmin = xmin - 0.1 * np.abs(xmin)
    xmax = xmax + 0.1 * np.abs(xmax)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))

    plt.plot([xmin, xmax], [xmin, xmax], color='black')
    plt.scatter(hankel_matrix[0, 1:], optimal_predictions, color='black')

    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Scatter plot for E = " + str(E) + r" and $\theta$ = " + str(optimal_theta))

    plt.show()

    return(optimal_theta)

#TODO: Make things parallel

if __name__ == "__main__":
    lorenz_trajectory = simulate_lorenz(t_max=1000, noise=0.01)
    lorenz_y = lorenz_trajectory[250:, 1]
    #white_noise = simulate_additive_white_noise(delta_t=2, t_max=1500, noise=2.5)

    #thomas_trajectory = simulate_thomas()
    #thomas_x = thomas_trajectory[:,0]

    #simplex_projection(lorenz_x)
    #plot_autocorrelation(lorenz_y)
    #simplex_projection(lorenz_y, lag = 181, max_E = 10)
    #S_map(lorenz_y, lag = 181, E = 3)

    #plot_embedding(lorenz_y, E = 3, lag = 181, filename = "")

    time_series = np.diff(lorenz_y)
    time_series = standardize_time_series(time_series)
    time_series = time_series[:, 0]

    #host_parasitoid = simulate_host_parasitoid(t_max=100)
    plot_time_series(time_series)
    #time_series = host_parasitoid[0]

    #plot_autocorrelation(time_series)
    #plot_partial_autocorrelation(time_series)
    #plot_recurrence(time_series[1:100], delay=8)
    #make_lag_scatterplot(time_series, lag=8)

    optimal_E = my_simplex_projection(time_series, lag=8, max_E=6, show_plots=False)
    #my_S_map(time_series, lag=8, E=optimal_E)
