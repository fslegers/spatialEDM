from deeptime import data
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from time_series_plots import *

def simulate_lorenz(vec0 = np.array([0, 1, 1.05]), delta_t = 2e-5, t_max = 100, noise = 0):
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

def simulate_thomas(vec0 = np.array([1, 1, 1]), delta_t = 2e-5, t_max = 100, noise = 0):
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
    system = data.thomas_attractor(h = delta_t)

    # integrate system from starting point
    trajectory = system.trajectory(vec0, t_max)

    # add random noise
    noise_array = np.random.normal(0.0, noise, size=trajectory.shape)
    trajectory = trajectory + noise_array

    return trajectory

def simulate_additive_white_noise(delta_t = 2e-5, t_max = 100, noise = 0.05):
    level = 0
    timer = 0
    trajectory = []
    while(timer < t_max):
        trajectory.append(level)
        level += np.random.normal(0, noise)
        timer += delta_t
    return np.array(trajectory)

def plot_dynamical_system(name = "Lorenz", which_var = 0,
                          delta_t = 2e-5, t_max = 1000, noise = 0.05,
                          tube_radius = 0.1, colors = "PuRd"):
    """
    Function that executes all plotting consecutively.
    :param name: determines which dynamical system is simulated.
    :param which_var: determines which variable of the system is plotted.
    :param delta_t: time steps of the system integration.
    :param t_max: how long the simulation runs.
    :param noise: determines how much noise is added to the trajectory.
    :param tube_radius: tube thickness in the 3D plot.
    :param colors: colors in the 3D plot.
    """
    if(name == "Lorenz"):
        trajectory = simulate_lorenz(vec0=np.array([1, 2, 3]), delta_t=delta_t, t_max=t_max, noise=noise)
        time_series = trajectory[:, which_var]
        make_3d_plot(trajectory[:,0], trajectory[:,1], trajectory[:,2],
                     filename="lorenz", tube_radius=tube_radius, colors=colors)

    elif(name == "Thomas"):
        trajectory = simulate_thomas(vec0=np.array([1, 2, 3]), delta_t=delta_t, t_max=t_max, noise=noise)
        time_series = trajectory[:, which_var]
        make_3d_plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                     filename="thomas", tube_radius=tube_radius, colors=colors)

    else:
        time_series = simulate_additive_white_noise(delta_t=2e-3, t_max=4, noise=0.1)
        time_series = np.ndarray.transpose(np.array(time_series))

    plot_time_series(time_series, filename=name)
    plot_autocorrelation(time_series, filename=name)
    #plot_recurrence(time_series, delay=1, eps=0.5, filename=name)
    plot_partial_autocorrelation(time_series, filename=name)

if __name__ == "__main__":
    #lorenz_trajectory = simulate_lorenz(t_max = 1000)
    #thomas_trajectory = simulate_thomas(vec0=np.array([1,2,3]),delta_t = 1, t_max = 50)
    #white_noise_trajectory = simulate_additive_white_noise(t_max = 200)

    #plot_correlation(lorenz_trajectory[:,0],lorenz_trajectory[:,1], window_size=10,filename="Lorenz xy")
    #plot_correlation(lorenz_trajectory[:,1], lorenz_trajectory[:,2], window_size=10, filename="Lorenz yz")
    #plot_correlation(lorenz_trajectory[:,0], lorenz_trajectory[:,2], window_size=10, filename="Lorenz xz")

    plot_dynamical_system(name = "Thomas", which_var = 0, delta_t = 0.02, t_max=1)

