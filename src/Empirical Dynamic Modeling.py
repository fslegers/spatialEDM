import math

import numpy as np
from pyEDM import *
import pandas as pd
from create_dummy_time_series import *
from sklearn import preprocessing
from scipy.stats import pearsonr

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

    # For each dimension E
    for dim in range(1, max_E + 1):
        # add original time series
        N = len(time_series) - dim*lag
        Hankel_matrix = []

        # add time delay embeddings
        for i in range(dim + 1):
            if i == 0:
                delayed_time_series = time_series[(dim - i)*lag:]
            else:
                delayed_time_series = time_series[(dim-i)*lag:-i*lag]
            Hankel_matrix.append(delayed_time_series)

        # create distance matrix
        Hankel_matrix = np.stack(Hankel_matrix, axis = 0)
        dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i, N):
                dist = np.linalg.norm((Hankel_matrix[:,i] - Hankel_matrix[:,j]))
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist

        # for all target points, get dim+1 nearest neighbors and make one-step-ahead prediction (weighted average)
        predictions = []

        for target in range(N):
            # find dim + 1 nearest neighbors
            nearest_neighbors = np.argpartition(dist_matrix[target,:], (0, dim + 2))
            nearest_neighbors = np.arange(N)[nearest_neighbors[1:dim+2]]

            min_distance = dist_matrix[target,nearest_neighbors[0]]
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
                    weight = np.exp(-dist_matrix[target,neighbor]/min_distance)
                    total_weight += weight
                    weighted_average += next_val * weight

                weighted_average = weighted_average/total_weight

            predictions.append(weighted_average)

            #TODO: in book, they have a minimum weight of 0.000001 (why?)

        if show_plots:
            plt.scatter(Hankel_matrix[0,:], predictions)
            plt.plot(range(0,int(max(time_series))), range(0,int(max(time_series))))
            plt.title("E = " + str(dim))
            plt.xlabel("Observed values", fontsize = 12)
            plt.ylabel("Predicted values", fontsize = 12)
            plt.show()

        # Pearson Correlation Coefficient
        cor = pearsonr(Hankel_matrix[0,:], predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(Hankel_matrix[0,:],predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(Hankel_matrix[0,:], predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

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

    return np.argmax(cor_list) + 1

def simplex_projection_replicates(time_series, lag = -1, max_E = 10):
    return 0

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
    # Embed time series
    N = len(time_series) - E*lag
    Hankel_matrix = []
    for i in range(E + 1):
        if i == 0:
            delayed_time_series = time_series[E * lag:]
        else:
            delayed_time_series = time_series[(E - i) * lag:-i * lag]
        Hankel_matrix.append(delayed_time_series)

    # create distance matrix
    Hankel_matrix = np.stack(Hankel_matrix, axis=0)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            dist = np.linalg.norm((Hankel_matrix[:, i] - Hankel_matrix[:, j]))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    cor_list = []
    mae_list = []
    rmse_list = []

    for theta in range(11):

        predictions = []

        for target in range(N):
            d_m = mean(np.concatenate((dist_matrix[target,:target], dist_matrix[target, target+1:])))

            if d_m == 0:
                print('Distance to all points is zero.')
                return 0

            weights = np.exp(-theta * dist_matrix[target, :] / d_m)
            weights[target] = 0
            next_val = np.dot(weights,np.transpose(time_series[1:N+1])) / sum(weights)
            predictions.append(next_val)

        # Pearson Correlation Coefficient
        cor = pearsonr(Hankel_matrix[0, :], predictions)[0]
        cor_list.append(cor)

        # Mean Absolute Error
        mae = mean(abs(np.subtract(Hankel_matrix[0, :], predictions)))
        mae_list.append(mae)

        # Root Mean Squared Error
        mse = mean(np.square(np.subtract(Hankel_matrix[0, :], predictions)))
        rmse = math.sqrt(mse)
        rmse_list.append(rmse)

    # Show figure of performance plots
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Performance measures per Theta')

    axs[0].set_ymargin(0.1)
    axs[1].set_ymargin(0.1)
    axs[2].set_ymargin(0.1)

    axs[0].plot(range(0, 11), cor_list, color = 'black', marker = 'o')
    axs[1].plot(range(0, 11), mae_list, color = 'black', marker = 'o')
    axs[2].plot(range(0, 11), rmse_list, color = 'black', marker = 'o')

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

    axs[0].plot(np.argmax(cor_list), max(cor_list), color='m', marker='D', markersize = 7)
    axs[1].plot(np.argmin(mae_list), min(mae_list), color='m', marker='D', markersize = 7)
    axs[2].plot(np.argmin(rmse_list), min(rmse_list), color='m', marker='D', markersize = 7)

    plt.show()

#TODO: put "embed time series" and "create distance matrix" into own functions
#TODO: Simplex plot predicted values against actual values for optimal E
#TODO: S-Map plot predicted values against actual values for optimal theta


if __name__ == "__main__":
    lorenz_trajectory = simulate_lorenz(t_max = 1000, noise = 1.5)
    lorenz_y = lorenz_trajectory[250:,1]

    white_noise = simulate_additive_white_noise(delta_t = 1, t_max = 300, noise = 0.5)

    #thomas_trajectory = simulate_thomas()
    #thomas_x = thomas_trajectory[:,0]

    #simplex_projection(lorenz_x)
    #plot_autocorrelation(lorenz_y)
    #simplex_projection(lorenz_y, lag = 181, max_E = 10)
    #S_map(lorenz_y, lag = 181, E = 3)

    #plot_embedding(lorenz_y, E = 3, lag = 181, filename = "")

    time_series = lorenz_y

    plot_time_series(time_series)
    plot_autocorrelation(time_series)
    plot_partial_autocorrelation(time_series)
    plot_recurrence(time_series[1:100], delay = 1)
    make_lag_scatterplot(time_series, lag = 1)

    my_simplex_projection(time_series, lag = 60, max_E = 5, show_plots = False)
    my_S_map(time_series, lag = 60, E = 5)