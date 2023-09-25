from src.create_dummy_time_series import simulate_lorenz
from src.time_series_plots import plot_autocorrelation
from empirical_dynamic_modeling import *
import random

def univariate_simulations(ts_length=100, process_noise=0.0, step_size=10, n_iter=10):

    counter = 0
    while counter < n_iter:
        # set initial values
        x_0 = random.uniform(-20, 20)
        y_0 = random.uniform(-30, 30)
        z_0 = random.uniform(0, 50)
        tau = 1

        vec0 = np.array([x_0, y_0, z_0])

        # sample lorenz attractor
        lorenz_traj = simulate_lorenz(vec0=vec0, t_max=100+ts_length*step_size, noise=process_noise)
        lorenz_traj = lorenz_traj[99:, 0] # should we delete first bit?

        # create less dense data set
        sparse_lorenz_traj = []
        for j in range(len(lorenz_traj)):
            if j % step_size == 0:
                sparse_lorenz_traj.append(lorenz_traj[j])

        # differentiate and standardize
        ts = np.diff(sparse_lorenz_traj)
        ts = standardize_time_series(ts)
        ts = ts[:, 0]

        plt.plot(np.arange(1, len(ts) + 1, 1), ts)
        plt.scatter(np.arange(1, len(ts) + 1, 1), ts)
        plt.title("Time series input")
        plt.show()

        # Perform EDM
        edm(ts, 1, 10, "CV", 4)

        counter += 1

    return 0


def multivariate_simulations(ts_length=25, n_time_series = 4, std_dev_init = 0.1, process_noise=0.1, step_size=10):

    ts_collection = []

    fig, axs = plt.subplots(n_time_series, sharex = True)

    counter = 0
    while counter < n_time_series:
        # set initial values
        x_0 = random.normalvariate(1.0, std_dev_init)
        y_0 = random.normalvariate(1.0, std_dev_init)
        z_0 = random.normalvariate(1.0, std_dev_init)

        vec0 = np.array([x_0, y_0, z_0])

        # sample lorenz attractor
        lorenz_traj = simulate_lorenz(vec0=vec0, t_max=ts_length * step_size, noise=process_noise)
        lorenz_traj = lorenz_traj[:, 0]

        # create less dense data set
        sparse_lorenz_traj = []
        for j in range(len(lorenz_traj)):
            if j % step_size == 0:
                sparse_lorenz_traj.append(lorenz_traj[j])

        # differentiate and standardize
        ts = np.diff(sparse_lorenz_traj)
        ts = standardize_time_series(ts)
        ts = ts[:, 0]

        # plot time series
        axs[counter].plot(np.arange(1, len(ts)+1, 1), ts)
        axs[counter].scatter(np.arange(1, len(ts) + 1, 1), ts)
        #axs[counter].set_title("Time series " + str(counter))

        ts_collection.append(ts)

        counter += 1

    fig.suptitle("Time series input")
    fig.show()

    # Perform EDM
    edm(ts_collection, 1, 10, "CV", 4)



if __name__ == "__main__":

    random.seed(123)

    print("#######   UNIVARIATE   #######")
    univariate_simulations(ts_length=25, process_noise = 0.1, n_iter = 1)

    print("#######   MULTIVARIATE   #######")
    multivariate_simulations(ts_length=8, n_time_series=4, process_noise= 0.1, std_dev_init=10)







