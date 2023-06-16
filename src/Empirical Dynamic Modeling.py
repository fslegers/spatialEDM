import numpy as np
from pyEDM import *
import pandas as pd
from create_dummy_time_series import *
from sklearn import preprocessing

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

def my_simplex_projection(time_series, lag = 1, max_E = 10):

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
            for j in range(len(time_series) - dim*lag - i):
                dist = np.linalg.norm(Hankel_matrix[:,i], Hankel_matrix[:,j], axis = 1)
                dist_matrix[i,j] = dist
                dist_matrix[j,i] = dist

        # for all target points, get dim+1 nearest neighbors and make one-step-ahead prediction (weighted average)
        #for target in range(len(time_series) - dim*lag):
            # find dim + 1 nearest neighbors



        # create embedding matrix
        embedding_matrix = time_series

    return 0



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

#TODO
#CCM
#Multi-view
#PredictInterval
#SpatialReplicates
#PlotEmbeddingSpace

if __name__ == "__main__":
    lorenz_trajectory = simulate_lorenz(t_max = 30000)
    lorenz_y = lorenz_trajectory[:,1]

    #thomas_trajectory = simulate_thomas()
    #thomas_x = thomas_trajectory[:,0]

    #simplex_projection(lorenz_x)
    #plot_autocorrelation(lorenz_y)
    #simplex_projection(lorenz_y, lag = 181, max_E = 10)
    #S_map(lorenz_y, lag = 181, E = 3)

    #plot_embedding(lorenz_y, E = 3, lag = 181, filename = "")

    time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    my_simplex_projection(time_series, lag = 1, max_E=4)