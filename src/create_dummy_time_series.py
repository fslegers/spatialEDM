from deeptime import data
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from time_series_plots import *

def simulate_lorenz(vec0 = np.array([1, 1, 1]), delta_t = 2e-5, t_max = 100, noise = 0):
    """
    Simulates a lorenz trajectory from initial point vec0, with time steps of size delta_t
    until time t_max.
    :type vec0: a numpy array of size 3
    :param vec0: vector that describes the initial state of the system
    :param delta_t: step size
    :param t_max: maximum time
    :param noise: amount of gaussian noise that is added to each coordinate of the trajectory
    :return: trajectory of x, y and z coordinates
    """
    # Check if vec0 is correct
    if isinstance(vec0, np.ndarray) and len(vec0) == 3:
        pass
    else:
        print("vec0 is not a np.array of size 3. Changing to [1,1,1].")
        vec0 = np.array([1, 1, 1])

    # initialize dynamical system
    system = data.lorenz_system(h = delta_t)

    # integrate system from starting point
    trajectory = system.trajectory(vec0, t_max)

    # add random noise
    noise_array = np.random.normal(0.0, noise, size = trajectory.shape)
    trajectory = trajectory + noise_array

    return trajectory

def simulate_thomas():
    """
        Simulates a thomas trajectory from initial point vec0, with time steps of size delta_t
        until time t_max.
        :type vec0: a numpy array of size 3
        :param vec0: vector that describes the initial state of the system
        :param delta_t: step size
        :param t_max: maximum time
        :param noise: amount of gaussian noise that is added to each coordinate of the trajectory
        :return: trajectory of x, y and z coordinates
        """
    # Check if vec0 is correct
    if isinstance(vec0, np.ndarray) and len(vec0) == 3:
        pass
    else:
        print("vec0 is not a np.array of size 3. Changing to [1,1,1].")
        vec0 = np.array([1, 1, 1])

    # initialize dynamical system
    system = data.thomas_attractor()

    # integrate system from starting point
    trajectory = system.trajectory(vec0, t_max)

    # add random noise
    noise_array = np.random.normal(0.0, noise, size=trajectory.shape)
    trajectory = trajectory + noise_array

    return trajectory

if __name__ == "__main__":
    # simulate a trajectory of the Lorenz system with noise
    lorenz_trajectory = simulate_lorenz(vec0 = np.array([1,2,3]), t_max = 25000, noise = 0.005)

    # sample the time series for the x coordinate
    lorenz_x = lorenz_trajectory[:, 0]

    # show plots for this time series
    plot_time_series(lorenz_x, filename="lorenz")
    plot_autocorrelation(lorenz_x, filename="lorenz")
    plot_recurrence(lorenz_x, delay = 2, eps = 0.5, filename="lorenz")

    # simulate a trajectory of the Thomas attractor
    thomas_trajectory = simulate_thomas(vec0=np.array([1, 2, 3]), t_max=25000, noise=0.005)

    # sample the time series for the x coordinate
    thomas_x = thomas_trajectory[:, 0]

    # show plots for this time series
    plot_time_series(thomas_x, filename="thomas")
    plot_autocorrelation(thomas_x, filename="thomas")
    plot_recurrence(thomas_x, delay=2, eps=0.5, filename="thomas")