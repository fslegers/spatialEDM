from src.time_series_plots import plot_autocorrelation
from empirical_dynamic_modeling import *
import random


def univariate_simulations(ts_length=100, noise=0.0, step_size=10, n_iter=10):

    counter = 0
    while counter <= n_iter:
        # set initial values
        x_0 = random.uniform(-20, 20)
        y_0 = random.uniform(-30, 30)
        z_0 = random.uniform(0, 50)
        tau = 1

        # sample lorenz attractor
        lorenz_traj = simulate_lorenz(t_max=750+ts_length*step_size, noise=noise)
        lorenz_traj = lorenz_traj[750:, 0]                                                 # should we delete first bit?

        # create less dense data set
        sparse_lorenz_traj = []
        for j in range(len(lorenz_traj)):
            if j % step_size == 0:
                sparse_lorenz_traj.append(lorenz_traj[j])

        # differentiate and standardize
        ts = np.diff(sparse_lorenz_traj)
        ts = standardize_time_series(ts)
        ts = ts[:, 0]

        # Perform EDM
        dim = simplex_projection(ts, lag=tau, max_E=sqrt(ts_length))
        smap(ts, lag=tau, E=dim)

        counter += 1

    return 0


def multivariate_simulations():
    return 0


if __name__ == "__main__":

    random.seed(123)

    univariate_simulations(ts_length=10)







