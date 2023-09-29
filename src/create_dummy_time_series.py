from time_series_plots import *
from scipy import integrate
from matplotlib import pyplot as plt
import numpy as np


def derivative_lv(X, t, alpha, beta, delta, gamma):
    x, y = X
    dotx = alpha * x - beta * x * y
    doty = -delta * y + gamma * x * y
    return np.array([dotx, doty])


def derivative_lorenz(X, t, sigma, rho, beta):
    x, y, z = X
    dotx = sigma*(y-x)
    doty = x*(rho-z)-y
    dotz = x*y - beta*z
    return np.array([dotx, doty, dotz])


def simulate_lotka_volterra(vec0 = np.array([4,2]),
                            alpha=1, beta=1, delta=1, gamma=1,
                            ntimesteps=1000, tmax=30,
                            obs_noise_sd=0):

    # TODO: add process noise?

    # Simulate trajectories of Lotka Volterra
    t = np.linspace(0, tmax, ntimesteps)
    res = integrate.odeint(derivative_lv, vec0, t, args = (alpha, beta, delta, gamma))
    x, y = res.T

    # Add obervation noise
    obs_noise = np.random.normal(loc=0.0, scale=obs_noise_sd, size=(2,len(x)))
    x += obs_noise[0,]
    y += obs_noise[1,]

    return(x,y,t)


def simulate_lorenz(vec0 = np.array([1,1,1]),
                      params = np.array([10, 28, 8.0/3.0]),
                      ntimesteps = 1000, tmax = 30,
                      obs_noise_sd=0):

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


def sample_from_ts(x, t, sampling_interval, n_points, spin_off=0):

    x_ = [x[i] for i in range(1, len(x)) if i % sampling_interval == 0 and i > spin_off]
    x = [x_[i] for i in range(1, len(x_)) if i <= n_points]

    t_ = [t[i] for i in range(1, len(t)) if i % sampling_interval == 0 and i > spin_off]
    t = [t_[i] for i in range(1, len(t_)) if i <= n_points]

    return x, t


def simulate_spatial_lorenz(init_points, init_params,
                           ntimesteps = 1000, tmax = 30,
                           obs_noise_sd=0):

    # TODO: Now, there has to be the same number of init_points as init_params
    # but we also want to be able to keep one constant and not the other

    n_plots = min(len(init_points), 4)
    fig, axs = plt.subplots(n_plots)

    all_trajectories = []

    for i in range(len(init_points)):

        init_point = init_points[i]
        init_param = init_params[i]

        trajectories_i = simulate_lorenz(vec0=init_point,
                                       params=init_param,
                                       ntimesteps=ntimesteps,
                                       tmax=tmax,
                                       obs_noise_sd=obs_noise_sd)

        trajectories_i.append(init_param)
        all_trajectories.append(trajectories_i)

        # Plot x variabele van de eerste 4 tijdreeksen
        if i < n_plots:
            label = "[" + "%0.2f" % init_point[0] + ", %0.2f" % init_point[1] + ", %0.2f" % init_point[2] + "] \n" + \
                    r"$\sigma$" + "= %0.2f" % init_param[0] + "\n" + r"$\rho$" + "= %0.2f" % init_param[1] + "\n" + \
                    r"$\beta$" + "= %0.2f" % init_param[2]
            axs[i].plot(trajectories_i[3], trajectories_i[0], label=label)
            axs[i].legend(loc=2, prop={'size': 5.9})

    for ax in axs[:-1]:
        ax.set_xticks([])

    fig.show()

    return all_trajectories


if __name__ == "__main__":

    init_points = [np.array([1,1,1])]
    init_params = [np.array([10, 28, 8.0/3.0])]

    for i in range(4):
        vec0 = np.random.uniform(0.0, 4.0, 3)

        # Code for uniformly distr. parameters
        #sigma0 = np.random.normal(10.0, 1.0)
        #rho0 = np.random.normal(28.0, 2.0)
        #beta0 = np.random.normal(8.0/3.0, 1.0)
        #param0 = np.array([sigma0, rho0, beta0])

        # Code for same parameters
        param0 = np.array([10, 28, 8.0/3.0])

        init_points.append(vec0)
        init_params.append(param0)

    simulate_spatial_lorenz(init_points, init_params)

    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
    #
    # x, y, z, t = simulate_lorenz(obs_noise_sd=0.0)
    # ax1.plot(t, x, label="x")
    # ax1.plot(t, y, label="y")
    # ax1.plot(t, z, label="z")
    # ax1.set_title(r'$\alpha$' + "= 0.0", verticalalignment='bottom')
    #
    # x, y, z, t = simulate_lorenz(obs_noise_sd=2.5)
    # ax2.plot(t, x, label="x")
    # ax2.plot(t, y, label="y")
    # ax2.plot(t, z, label="z")
    # ax2.set_title(r'$\alpha$' + "= 2.5", verticalalignment='bottom')
    #
    # x, y, z, t = simulate_lorenz(obs_noise_sd=5.0)
    # ax3.plot(t, x, label="x")
    # ax3.plot(t, y, label="y")
    # ax3.plot(t, z, label="z")
    # ax3.set_title(r'$\alpha$' + "= 5.0", verticalalignment='bottom')
    #
    # x, y, z, t = simulate_lorenz(obs_noise_sd=7.5)
    # ax4.plot(t, x, label="x")
    # ax4.plot(t, y, label="y")
    # ax4.plot(t, z, label="z")
    # ax4.set_title(r'$\alpha$' + "= 7.5", verticalalignment='bottom')
    #
    # for ax in fig.get_axes():
    #     ax.label_outer()
    #
    # fig.suptitle("Lorenz system with observation noise", verticalalignment='top')
    # fig.tight_layout()
    #
    # ##############################################################################
    #
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    #
    # x, y,t = simulate_lotka_volterra(obs_noise_sd=0.0)
    # ax1.plot(t, x, label="x")
    # ax1.plot(t, y, label="y")
    # ax1.set_title(r'$\alpha$' + "= 0.0", verticalalignment='bottom')
    #
    # x, y, t = simulate_lotka_volterra(obs_noise_sd=0.25)
    # ax2.plot(t, x, label="x")
    # ax2.plot(t, y, label="y")
    # ax2.set_title(r'$\alpha$' + "= 0.25", verticalalignment='bottom')
    #
    # x, y, t = simulate_lotka_volterra(obs_noise_sd=0.5)
    # ax3.plot(t, x, label="x")
    # ax3.plot(t, y, label="y")
    # ax3.set_title(r'$\alpha$' + "= 0.5", verticalalignment='bottom')
    #
    # x, y, t = simulate_lotka_volterra(obs_noise_sd=1.5)
    # ax4.plot(t, x, label="x")
    # ax4.plot(t, y, label="y")
    # ax4.set_title(r'$\alpha$' + "= 1.5", verticalalignment='bottom')
    #
    # for ax in fig.get_axes():
    #     ax.label_outer()
    #
    # fig.suptitle("Lotka Volterra with observation noise", verticalalignment='top')
    # fig.tight_layout()
    #
    # fig.show()


