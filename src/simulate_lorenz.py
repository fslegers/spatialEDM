from scipy import integrate
from matplotlib import pyplot as plt
import random
import numpy as np

def derivative_lorenz(X, t, sigma, rho, beta):
    x, y, z = X
    dotx = sigma*(y-x)
    doty = x*(rho-z)-y
    dotz = x*y - beta*z
    return np.array([dotx, doty, dotz])


def simulate_lorenz(vec0 = np.array([1,1,1]),
                    params = np.array([10, 28, 8.0/3.0]),
                    ntimesteps = 1000, tmax = 30,
                    obs_noise_sd=0.0):

    # TODO: add process noise?

    t = np.linspace(0, tmax, ntimesteps)
    res = integrate.odeint(derivative_lorenz, vec0, t, args=(params[0], params[1], params[2]))
    x, y, z = res.T

    # Add obervation noise
    obs_noise = np.random.normal(loc=0.0, scale=obs_noise_sd, size=(3,len(x)))
    x += obs_noise[0,]
    y += obs_noise[1,]
    z += obs_noise[2,]

    return [x,y,z,t]


def sample_from_ts(x, t, sampling_interval, n_points=-1, spin_off=0, sample_end=False):

    x_ = [x[i] for i in range(len(x)) if i % sampling_interval == 0 and i > spin_off]
    t_ = [t[i] for i in range(len(t)) if i % sampling_interval == 0 and i > spin_off]

    # Take first n_points observations
    if not sample_end and n_points >= 1:
        x = [x_[i] for i in range(len(x_)) if i <= n_points]
        t = [t_[i] for i in range(len(t_)) if i <= n_points]

    # Take last n_points observations
    if sample_end and n_points >= 1:
        x = [x_[i] for i in range(len(x_)) if i >= len(x_) - n_points]
        t = [t_[i] for i in range(len(t_)) if i >= len(t_) - n_points]

    if n_points < 1:
        x = x_
        t = t_

    return x, t



def simulate_spatial_lorenz(initial_vec,
                            dt_initial,
                            initial_params,
                            delta_rho=0.0,
                            std_noise=0.0,
                            n_ts=5,
                            ts_length=10,
                            ts_interval=10):
    """
    First, a long trajectory of the Lorenz system is simulated for each combination of initial parameters. From these,
    n_ts shorter time series with possibly larger sampling intervals are sampled. Then, observational noise is added.
    """

    if dt_initial > 0 and delta_rho > 0:
        print("It is not possible to change both the initial coordinates and the parameter rho at the moment. " \
        "Changing dt_initial_coord to 0")
        dt_initial = 0

    if dt_initial <= 0 and delta_rho <= 0:
        print("At least one of dt_initial_coord and delta_rho has to be positive. Changing dt_initial_coord to 1")
        delta_rho = 0
        dt_initial = 1

    # Save all trajectories in one place
    all_trajectories = []
    parameter_values = []

    if dt_initial > 0:
        string = r'$t_0$ = '
        # Simulate one long time series
        x_long, y_long, z_long, t_long = simulate_lorenz(vec0=initial_vec,
                                                         params=initial_params,
                                                         ntimesteps=max(1000, ts_length*ts_interval*(n_ts+1)),
                                                         tmax=30,
                                                         obs_noise_sd=std_noise)
        del(y_long, z_long)

        # From this time series, sample multiple short time series
        # with starting conditions dt_initial away from each other
        for i in range(0, n_ts):
            x, t = sample_from_ts(x_long[dt_initial * i:], t_long[dt_initial * i:],
                                              sampling_interval=ts_interval,
                                              n_points=ts_length)
            all_trajectories.append(x)
            parameter_values.append(dt_initial*i)

    elif delta_rho > 0:
        string = "ρ = "
        # For each different value of rho, simulate a time series
        for i in range(0, n_ts):
            x, y, z, t = simulate_lorenz(vec0=initial_vec,
                                         params=initial_params,
                                         ntimesteps=max(1000, ts_length*(ts_interval+1)),
                                         tmax=30,
                                         obs_noise_sd=std_noise)

            x, t = sample_from_ts(x, t, sampling_interval=ts_interval, n_points=ts_length)
            all_trajectories.append(x)

            # change initial rho
            parameter_values.append(initial_params[1])
            initial_params[1] += delta_rho

    # Prepare to make some plot of some time series
    n_plots = min(n_ts, 5)
    which_to_plot = random.sample(list(np.arange(0,n_ts)), n_plots)
    which_to_plot.sort()
    fig, axs = plt.subplots(n_plots)

    for i in range(0, n_plots):
        x = all_trajectories[which_to_plot[i]]
        axs[i].plot(np.arange(0, len(x)), x, label=string+str(parameter_values[which_to_plot[i]]))
        axs[i].legend(loc='upper right', handlelength=0)

    for ax in axs[:-1]:
        ax.set_xticks([])

    fig.show()

    return all_trajectories


if __name__ == "__main__":

    x, y, z, t = simulate_lorenz(vec0=[4.548120346844322,-2.081443690988742,30.804556029728243], tmax=8, ntimesteps=350)

    plt.plot(t, x, color='blue')
    plt.plot(t, y, color='green')
    plt.plot(t, z, color='orange')
    plt.ylim((-21, 41))
    plt.xlabel("time")
    plt.show()

    plt.plot(t, x, color='blue')
    plt.xlabel("time")
    plt.ylim((-21, 41))
    plt.show()




