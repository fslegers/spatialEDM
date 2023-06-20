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
        Hankel_matrix = []

        # add time delay embeddings
        for i in range(dim + 1):
            if i == 0:
                delayed_time_series = time_series[(dim - i) * lag:]
            else:
                delayed_time_series = time_series[(dim-i)*lag:-i*lag]
            Hankel_matrix.append(delayed_time_series)

        # create distance matrix
        Hankel_matrix = np.stack(Hankel_matrix, axis = 0)
        dist_matrix = np.zeros((len(time_series) - dim*lag, len(time_series) - dim*lag))
        for i in range(len(time_series) - dim*lag):
            for j in range(i, len(time_series) - dim*lag):
                dist = np.linalg.norm((Hankel_matrix[:,i] - Hankel_matrix[:,j]))
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist

        # for all target points, get dim+1 nearest neighbors and make one-step-ahead prediction (weighted average)
        predictions = []
        N = len(time_series) - dim*lag

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
            plt.scatter(pearsonr(Hankel_matrix[0,:], predictions)[0], predictions)
            plt.plot(range(0,int(max(time_series))), range(0,int(max(time_series))))
            plt.title("E = " + str(dim + 1))
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
    axs[0].plot(range(2,len(cor_list)+2), cor_list)
    axs[1].plot(range(2, len(mae_list)+2), mae_list)
    axs[2].plot(range(2, len(rmse_list)+2), rmse_list)
    axs[0].set_ylabel('rho')
    axs[1].set_ylabel('MAE')
    axs[2].set_ylabel('RMSE')
    axs[2].set_xlabel('E')
    axs[0].set_ylim(min(cor_list), 1)
    axs[0].xaxis.grid(True)
    axs[1].xaxis.grid(True)
    axs[2].xaxis.grid(True)
    plt.show()


    print("Highest correlation for E = :", str(np.argmax(cor_list) + 2) + " (" + str(max(cor_list)) + ")")
    print("Lowest MAE for E = :", str(np.argmin(mae_list) + 2) + " (" + str(min(mae_list)) + ")")
    print("Lowest RMSE for E = :", str(np.argmin(rmse_list) + 2) + " (" + str(min(rmse_list)) + ")")

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
    fig.suptitle('Performance measures per E')
    axs[0].plot(range(0, 11), cor_list)
    axs[1].plot(range(0, 11), mae_list)
    axs[2].plot(range(0, 11), rmse_list)
    axs[0].set_ylabel('rho')
    axs[1].set_ylabel('MAE')
    axs[2].set_ylabel('RMSE')
    axs[2].set_xlabel('theta')
    axs[0].set_ylim(min(cor_list), 1)
    axs[0].xaxis.grid(True)
    axs[1].xaxis.grid(True)
    axs[2].xaxis.grid(True)
    plt.show()



#TODO: put "embed time series" and "create distance matrix" into own functions


if __name__ == "__main__":
    lorenz_trajectory = simulate_lorenz(t_max = 300, noise = 0.5)
    lorenz_y = lorenz_trajectory[:,1]

    #thomas_trajectory = simulate_thomas()
    #thomas_x = thomas_trajectory[:,0]

    #simplex_projection(lorenz_x)
    #plot_autocorrelation(lorenz_y)
    #simplex_projection(lorenz_y, lag = 181, max_E = 10)
    #S_map(lorenz_y, lag = 181, E = 3)

    #plot_embedding(lorenz_y, E = 3, lag = 181, filename = "")

    time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    my_simplex_projection(time_series, lag = 1, max_E=5, show_plots=False)
    my_S_map(time_series, lag = 1, E = 2)