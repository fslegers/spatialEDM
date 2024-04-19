import time

import matplotlib.pyplot as plt

from src.classes import *
from src.simulate_lorenz import *
from multiprocessing import Pool
from itertools import product
from functools import partial
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def sample_lorenz(vec_0, params, n_points, obs_noise, sampling_interval=5):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=sampling_interval, n_points=n_points)
    t = [i for i in range(len(t))]
    return x, t


def sample_initial_points(n, rho):
    np.random.seed(1)

    # Generate one giant trajectory
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 2000, 30, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # Sample multiple initial points from it
    initial_vecs = []
    indices = np.random.randint(len(x), size=n)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    return initial_vecs


def sample_trajectory(obs_noise, length, vec0, rho):

    x, t = sample_lorenz(vec0, [10, rho, 8/3], length, obs_noise)
    x, preprocessing_info = preprocessing(x, t, 0)

    return x, t, preprocessing_info


def make_plot(x, t, preprocessing_info, obs, pred, time_stamps, rho, training_length, obs_noise, iter):

    mean = preprocessing_info[1]['mean']
    std_dev = preprocessing_info[1]['std_dev']
    trend = preprocessing_info[1]['trend']

    # Reverse preprocessing
    x = reverse_normalization(x, mean, std_dev)
    x = add_linear_trend(x, trend, t)

    obs = reverse_normalization(np.array(obs), mean, std_dev)
    obs = add_linear_trend(obs, trend, time_stamps)

    pred = reverse_normalization(np.array(pred), mean, std_dev)
    pred = add_linear_trend(pred, trend, time_stamps)

    # Select last bit
    x = x[-(15 + len(obs)):]
    t = t[-(15 + len(obs)):]

    plt.plot(t, x, color='grey', zorder=0, linewidth=1)
    plt.scatter(t, x, color='grey', zorder=1, s=25)

    plt.plot(time_stamps, pred, color='black', zorder=0, linewidth=1)
    for i in range(1, len(pred)+1):
        plt.scatter(time_stamps[i - 1], pred[i - 1], label=11-i, zorder=1, s=50)

    title = f"rho = {rho}, training length = {training_length}, noise = {obs_noise}"
    plt.title(title)
    plt.legend(title='horizon', loc="lower left", ncol = 5, bbox_to_anchor=(0.15, -0.5))
    plt.xlabel('t')
    plt.ylabel('x(t)')

    plt.tight_layout(pad=2)

    path = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/observed + predicted plots/{title}, {iter}.png"
    plt.savefig(path)

    plt.show()


def do_EDM(obs_noise, training_length, max_horizon, rho, n_iterations):

    initial_vecs = sample_initial_points(n_iterations, rho)

    for i in range(n_iterations):
        vec = initial_vecs[i]

        # Sample a time series and perform preprocessing
        x, t, preprocessing_info = sample_trajectory(obs_noise, training_length + 10, vec, rho)

        # Put them together into one library
        collection = []
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "", 0))

        # Split train and test set
        ts_train = [point for point in collection if point.time_stamp < training_length]
        ts_test = [point for point in collection if point.time_stamp >= training_length]
        del collection

        # Train model and predict test set
        model = EDM()
        model.train(ts_train, max_dim=3)
        _, smap = model.predict(ts_test, hor=max_horizon)

        # Measure performance
        smap = smap.dropna()

        # # Make scatter plots of obs vs pred
        # for hor in range(1, 11):
        #     plt.scatter(smap[smap['hor'] == hor]['obs'], smap[smap['hor'] == hor]['pred'], label=hor)
        # plt.xlabel('Observations')
        # plt.ylabel('Predictions')
        # plt.legend()
        # plt.show()

        predictions = []
        observations = []
        time_stamps = []

        horizon = 10
        time_stamp = smap[smap['hor']==10]['time_stamp'].values[0]

        while horizon >= 1:
            predictions.append(smap[(smap['hor']==horizon) & (smap['time_stamp']==time_stamp)]['pred'].values[0])
            observations.append(smap[(smap['hor']==horizon) & (smap['time_stamp']==time_stamp)]['obs'].values[0])
            time_stamps.append(time_stamp)
            horizon -= 1
            time_stamp -= 1

        make_plot(x, t, preprocessing_info, observations, predictions, time_stamps, rho, training_length, obs_noise, i)


if __name__ == "__main__":

    for rho in [20, 28]:
        for obs_noise in [0.0, 2.0, 4.0]:
            for training_length in [25, 75]:
                do_EDM(obs_noise, training_length, 10, rho, 3)





