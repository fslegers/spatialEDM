from src.classes import *
from src.simulate_lorenz import *
from multiprocessing import Pool
from itertools import product
from functools import partial
import warnings

import os

warnings.simplefilter(action='ignore', category=FutureWarning)

def sample_lorenz(vec_0, params, n_points, obs_noise, sampling_interval=5):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=sampling_interval, n_points=n_points)
    t = [i for i in range(len(t))]
    return x, t


def sample_initial_points(n, rho):
    # Generate one giant trajectory
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 2000, 30, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # Sample multiple initial points from it
    initial_vecs = []
    indices = np.random.randint(len(x), size=n)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    return initial_vecs


def sample_trajectory(obs_noise, length, vec0, rho, sampling_interval=5):
    x, t = sample_lorenz(vec0, [10, rho, 8/3], length, obs_noise, sampling_interval)
    x, preprocessing_info = preprocessing(x, t, 0)

    return x, t, preprocessing_info


def calculate_performance(result, preprocessing_info):
    # Reverse preprocessing
    result = reverse_preprocessing(result, preprocessing_info)

    # Calculate performance measures
    diff = np.subtract(result['obs'], result['pred'])
    performance = {}
    performance['MAE'] = np.mean(abs(diff))
    performance['RMSE'] = math.sqrt(np.mean(np.square(diff)))

    try:
        performance['corr'] = pearsonr(result['obs'], result['pred'])[0]
    except:
        performance['corr'] = None

    return performance


def loop_over_arguments(argument, max_horizon, rho, n_iterations):
    np.random.seed(123)
    obs_noise = argument[0]
    training_length = argument[1]

    initial_vecs = sample_initial_points(n_iterations, rho)
    results = pd.DataFrame()

    for i in range(n_iterations):
        vec = initial_vecs[i]

        # Sample a time series and perform preprocessing
        x, t, preprocessing_info = sample_trajectory(obs_noise, training_length + 25, vec, rho)

        # Put them together into one library
        collection = []
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "", 0))
        del x, t

        # Split train and test set
        ts_train = [point for point in collection if point.time_stamp < training_length]
        ts_test = [point for point in collection if point.time_stamp >= training_length]
        del collection

        # Train model and predict test set
        model = EDM(max_dim=max(int(np.sqrt(training_length)), 10))
        model.train(ts_train)
        _, smap = model.predict(ts_test, hor=max_horizon)

        # Measure performance
        smap = smap.dropna()

        for hor in range(1, max_horizon + 1):
            result_hor = reverse_preprocessing(smap[smap['hor'] == hor], [preprocessing_info])
            result_hor = pd.DataFrame(result_hor).drop(['location', 'species'], axis=1)
            result_hor['hor'] = hor
            result_hor['iter'] = i
            results = pd.concat([results, result_hor], axis=0)

    file_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/"
                 f"rho = {rho}/training_length = {training_length}, noise = {obs_noise}.csv")
    results.to_csv(file_name, index=False)


def loop_over_sampling_intervals(argument, max_horizon, rho, n_iterations):
    np.random.seed(123)
    obs_noise = argument[0]
    training_length = argument[1]
    sampling_interval = argument[2]

    initial_vecs = sample_initial_points(n_iterations, rho)
    results = pd.DataFrame()

    for i in range(n_iterations):
        vec = initial_vecs[i]

        # Sample a time series and perform preprocessing
        x, t, preprocessing_info = sample_trajectory(obs_noise, training_length + 25, vec, rho, sampling_interval)

        # Put them together into one library
        collection = []
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "", 0))
        del x, t

        # Split train and test set
        ts_train = [point for point in collection if point.time_stamp < training_length]
        ts_test = [point for point in collection if point.time_stamp >= training_length]
        del collection

        # Train model and predict test set
        model = EDM(max_dim=max(int(np.sqrt(training_length)), 10))
        model.train(ts_train)
        _, smap = model.predict(ts_test, hor=max_horizon)

        # Measure performance
        smap = smap.dropna()

        for hor in range(1, max_horizon + 1):
            result_hor = reverse_preprocessing(smap[smap['hor'] == hor], [preprocessing_info])
            result_hor = pd.DataFrame(result_hor).drop(['location', 'species'], axis=1)
            result_hor['hor'] = hor
            result_hor['sampling_interval'] = sampling_interval
            result_hor['iter'] = i
            results = pd.concat([results, result_hor], axis=0)

    file_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/single time series/"
                 f"sampling interval/rho = {rho}/training_length = {training_length}, noise = {obs_noise}, interval = {sampling_interval}.csv")
    results.to_csv(file_name, index=False)


def do_multiprocessing(func, argument_list):
    with Pool(8) as pool:
        pool.imap(func=func, iterable=argument_list)
        pool.close()
        pool.join()


def repeatedly_do_EDM(n_iterations=100, max_horizon=1, rho=28):
    noise_levels = [0.0, 2.0, 4.0]
    training_length = [25, 50, 75, 100, 125, 150]

    # Select this partial function for EDM with horizon and training_length
    #argument_list = product(noise_levels, training_length)
    #partial_function = partial(loop_over_arguments, max_horizon=max_horizon, rho=rho, n_iterations=n_iterations)

    # Select this partial function for the test RMSE per sampling interval:
    sampling_intervals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    argument_list = product(noise_levels, training_length, sampling_intervals)
    partial_function = partial(loop_over_sampling_intervals, max_horizon=max_horizon, rho=rho, n_iterations=n_iterations)

    do_multiprocessing(func=partial_function, argument_list=argument_list)


if __name__ == "__main__":
    rho = 28
    repeatedly_do_EDM(n_iterations=1, max_horizon=10, rho=rho)

    rho = 20
    repeatedly_do_EDM(n_iterations=1, max_horizon=10, rho=rho)





