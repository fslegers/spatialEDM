import math

import numpy as np
from pyEDM import *
import pandas as pd
from create_dummy_time_series import *
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

def my_simplex_projection(time_series, lag = 1, max_E = 10, show_plots = False):
    #TODO: think about embedding dimension: number of lags or dimension of state space?
    cor_list = []
    mae_list = []
    rmse_list = []

    optimal_cor = 0
    optimal_predictions = []

    # For each dimension E
    for dim in range(1, max_E + 1):
        Hankel_matrix = create_hankel_matrix(time_series, lag, dim)
        dist_matrix = create_distance_matrix(Hankel_matrix)

        # for all target points, get dim+1 nearest neighbors and make one-step-ahead prediction (weighted average)
        predictions = []
        N = Hankel_matrix.shape[1]

        for target in range(N-1):
            # find dim + 1 nearest neighbors
            nearest_neighbors = np.argpartition(dist_matrix[target, :], (0, dim + 2))
            nearest_neighbors = np.arange(N)[nearest_neighbors[1:dim+2]]

            min_distance = dist_matrix[target, nearest_neighbors[0]]
            weighted_average = 0
            total_weight = 0

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

                weighted_average = weighted_average/total_weight

            predictions.append(weighted_average)

            #TODO: in book, they have a minimum weight of 0.000001 (why?)

        if show_plots:
            plt.scatter(Hankel_matrix[0, :], predictions)
            plt.plot(range(0,int(max(time_series))), range(0,int(max(time_series))))
            plt.title("E = " + str(dim))
            plt.xlabel("Observed values", fontsize = 12)
            plt.ylabel("Predicted values", fontsize = 12)
            plt.show()

        # Pearson Correlation Coefficient
        cor = pearsonr(Hankel_matrix[0, 1:], predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(Hankel_matrix[0, 1:], predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(Hankel_matrix[0, 1:], predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

        if cor >= optimal_cor:
            optimal_cor = cor
            optimal_predictions = predictions
            optimal_E = dim

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

    major_tick = range(1,max_E + 1)
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
    Hankel_matrix = create_hankel_matrix(time_series, lag, E = optimal_E)
    xmin = min(min(optimal_predictions), min(Hankel_matrix[0, 1:]))
    xmax = max(max(optimal_predictions), max(Hankel_matrix[0, 1:]))

    xmin = xmin - 0.1 * np.abs(xmin)
    xmax = xmax + 0.1 * np.abs(xmax)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))

    plt.plot([xmin, xmax], [xmin, xmax], color='black')
    plt.scatter(Hankel_matrix[0, 1:], optimal_predictions, color='black')

    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Simplex results for E = " + str(np.argmax(cor_list) + 1))

    plt.show()

    return optimal_E

def S_map(time_series, lag = -1, E = 10):
    """
    Evaluates Smap prediction skill for different values of localization parameter theta)
    :param time_series:
    :param lag:
    :param E:
    :return:
    """
    #TODO
    #Add check if time_series is standardized
    
    # If no observation times are given, add them to time_series
    if len(np.shape(time_series)) == 1:
        obs_times = np.arange(1, np.shape(time_series)[0] + 1, 1)
        time_series = np.column_stack((obs_times, time_series))

    # Turn time_series into pandas dataframe (column vector)
    df = pd.DataFrame(time_series, columns = ["t", "x"])

    # Divide time series in training (60%), and test set (40%)
    length = len(df)
    training_set = "1 " + str(floor(0.6 * length))
    test_set = str(floor(0.6 * length) + 1) + " " + str(length)

    # Evaluate SMap prediction skill for localization parameter theta
    rho_per_theta = PredictNonlinear(dataFrame = df, E = E, tau = -np.abs(lag),
                                     lib = training_set, pred = test_set, columns = "x", embedded = False)

    return(rho_per_theta)

def my_S_map(time_series, lag = 1, E = 2):
    Hankel_matrix = create_hankel_matrix(time_series, lag, E)
    dist_matrix = create_distance_matrix(Hankel_matrix)

    N = Hankel_matrix.shape[1]
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
            d_m = mean(np.concatenate((dist_matrix[target, :target], dist_matrix[target, target+1:])))

            if d_m == 0:
                print('Distance to all points is zero.')
                return 0

            weights = np.exp(-theta * dist_matrix[target, :] / d_m)
            weights[target] = 0
            next_val = np.dot(weights, np.transpose(time_series[1:N+1])) / sum(weights)
            predictions.append(next_val)

        # Pearson Correlation Coefficient
        cor = pearsonr(Hankel_matrix[0, 1:], predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(Hankel_matrix[0, 1:], predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(Hankel_matrix[0, 1:], predictions)))
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
    xmin = min(min(optimal_predictions), min(Hankel_matrix[0, 1:]))
    xmax = max(max(optimal_predictions), max(Hankel_matrix[0, 1:]))

    xmin = xmin - 0.1 * np.abs(xmin)
    xmax = xmax + 0.1 * np.abs(xmax)
    plt.xlim((xmin, xmax))
    plt.ylim((xmin, xmax))

    plt.plot([xmin, xmax], [xmin, xmax], color='black')
    plt.scatter(Hankel_matrix[0, 1:], optimal_predictions, color='black')

    plt.xlabel("Observed values")
    plt.ylabel("Predicted values")
    plt.title("Scatter plot for E = " + str(E) + r" and $\theta$ = " + str(optimal_theta))

    plt.show()

    return(optimal_theta)

#TODO: Make things parallel

if __name__ == "__main__":
    #lorenz_trajectory = simulate_lorenz(t_max=1000, noise=0)
    #lorenz_y = lorenz_trajectory[250:, 1]

    white_noise = simulate_additive_white_noise(delta_t=2, t_max=1500, noise=2.5)

    #thomas_trajectory = simulate_thomas()
    #thomas_x = thomas_trajectory[:,0]

    #simplex_projection(lorenz_x)
    #plot_autocorrelation(lorenz_y)
    #simplex_projection(lorenz_y, lag = 181, max_E = 10)
    #S_map(lorenz_y, lag = 181, E = 3)

    #plot_embedding(lorenz_y, E = 3, lag = 181, filename = "")

    time_series = white_noise
    #time_series = lorenz_y

    plot_time_series(time_series)
    # plot_autocorrelation(time_series)
    # plot_partial_autocorrelation(time_series)
    # plot_recurrence(time_series[1:100], delay = 1)
    # make_lag_scatterplot(time_series, lag = 1)

    optimal_E = my_simplex_projection(time_series, lag=1, max_E=10, show_plots=False)
    my_S_map(time_series, lag=1, E=optimal_E)
