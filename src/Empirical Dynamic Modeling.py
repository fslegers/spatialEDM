from pyEDM import *
import pandas as pd
from create_dummy_time_series import *

def simplex_projection(time_series, time_interval = 1, lag = -1, max_E = 10):
    """
    Finds the optimal value for the embedding dimension E by one-step-ahead predictions
    using E+1 Nearest Neighbors.
    :return: E, optimal dimension
    """
    # If no observation times are given, add them to time_series
    if len(np.shape(time_series)) == 1:
        obs_times = np.arange(0, np.shape(time_series)[0] * time_interval , time_interval)
        time_series = np.column_stack((obs_times, time_series))

    # Turn time_series into pandas dataframe
    df = pd.DataFrame(time_series, columns = ["t", "x"])

    # Divide time series in training (60%), and test set (40%)
    length = len(df)
    training_set = "1 " + str(floor(0.6 * length))
    test_set = str(floor(0.6 * length) + 1) + " " + str(length)

    # Plot and return prediction skill rho for each embedding via Simplex
    rho_per_E = EmbedDimension(dataFrame = df, Tp = time_interval, maxE = max_E, tau = -np.abs(lag),
                               lib = training_set, pred = test_set, columns = "x")

    # Find the optimal E
    optimal_param = rho_per_E.loc[rho_per_E['rho'].idxmax()]
    print("Optimal embedding dimension E: ", str(optimal_param["E"]),
          " ( rho = ", str(optimal_param["rho"]), ").")

    return(optimal_param["E"])

def S_map(time_series, lag = -1, E = 10):
    """
    Evaluates Smap prediction skill for different values of localization parameter theta)
    :param time_series:
    :param lag:
    :param E:
    :return:
    """
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
#S-Map
#CCM
#Multi-view
#PredictInterval
#PredictNonlinear
#SpatialReplicates

if __name__ == "__main__":
    lorenz_trajectory = simulate_lorenz()
    lorenz_x = lorenz_trajectory[:,0]

    #thomas_trajectory = simulate_thomas()
    #thomas_x = thomas_trajectory[:,0]

    #simplex_projection(lorenz_x)
    plot_autocorrelation(lorenz_x)
    simplex_projection(lorenz_x, time_interval = 2e-5, lag = -37)
    #S_map(lorenz_x, lag = -37)