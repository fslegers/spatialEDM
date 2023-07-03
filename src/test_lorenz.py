from empirical_dynamic_modeling import *
import random

from src.time_series_plots import plot_autocorrelation

if __name__ == "__main__":

    n_iter  = 3
    max_noise = 0.5
    sparsity = 10 # lower value -> denser dataset
    which_var = 0 # choose x variable
    random.seed(123)

    counter = 0
    while counter <= n_iter:
        # set parameters
        x_0 = random.uniform(0, 1)
        y_0 = random.uniform(0, 1)
        z_0 = random.uniform(0, 1)
        noise_iter = np.linspace(0, max_noise, n_iter + 1)[counter]

        # sample lorenz attractor
        lorenz_traj = simulate_lorenz(t_max=2500, noise=noise_iter)
        lorenz_traj = lorenz_traj[750:, which_var]

        # create less dense data set
        sparse_lorenz_traj = []
        for i in range(len(lorenz_traj)):
            if i % sparsity == 0:
                sparse_lorenz_traj.append(lorenz_traj[i])

        # differentiate and standardize
        time_series = np.diff(sparse_lorenz_traj)
        time_series = standardize_time_series(time_series)
        time_series = time_series[:, 0]

        # Plot time series
        plot_time_series(time_series)
        optimal_lag = plot_autocorrelation(time_series)

        # Perform EDM
        optimal_E = my_simplex_projection(time_series, lag= optimal_lag, max_E=10)
        #plot_embedding(time_series, E= min(2, optimal_E), lag = optimal_lag)
        my_S_map(time_series, lag=optimal_lag, E=optimal_E)

        counter += 1





