import numpy as np

from src.classes import *
from src.simulate_lorenz import *
from multiprocessing import Pool
from itertools import product
from functools import partial
import warnings
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

def sample_lorenz(vec_0, params, n_points, obs_noise):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i for i in range(len(t))]
    return x, t


def sample_initial_points(n):
    # Generate one giant trajectory
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8 / 3], 2000, 30, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # Sample multiple initial points from it
    initial_vecs = []
    indices = np.random.randint(len(x), size=n)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    return initial_vecs


def sample_multiple_rhos(vec_0, n_points, n_repl, obs_noise, var):
    list_x, list_t, list_preprocessing = [], [], []

    i = 0
    while i < n_repl:
        rho = np.random.normal(28, var)
        x, t = sample_lorenz(vec_0, [10, rho, 8/3], n_points, obs_noise)

        # Preprocessing
        x, preprocessing_info = preprocessing(x, t, loc=i)
        list_preprocessing.append(preprocessing_info)

        list_x.append(x)
        list_t.append(t)
        i += 1

    return list_x, list_t, list_preprocessing


def sample_multiple_initial_values(vec_0, n_points, n_repl, obs_noise, var):
    list_x, list_t, list_preprocessing = [], [], []

    i = 0
    while i < n_repl:
        x_0 = np.random.normal(vec_0[0], var)
        y_0 = np.random.normal(vec_0[1], var)
        z_0 = np.random.normal(vec_0[2], var)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, 28, 8/3], n_points, obs_noise)

        # Preprocessing
        x, preprocessing_info = preprocessing(x, t, loc=i)
        list_preprocessing.append(preprocessing_info)

        list_x.append(x)
        list_t.append(t)
        i += 1

    return list_x, list_t, list_preprocessing


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


def loop_over_noise_levels(argument, n_replicates, training_length, max_horizon, test, rho):
    if n_replicates == 0:
        obs_noise = argument[1][0]
        variance = 0.0
        vec = argument[1][1]
        iter = argument[0]
    else:
        obs_noise = argument[1][0]
        variance = argument[1][1]
        vec = argument[1][2]
        iter = argument[0]

    # Sample replicate time series
    if test == "begin_conditions":
        xs, ts, preprocessing_info = sample_multiple_initial_values(vec, training_length + 25, n_replicates, obs_noise,
                                                                    variance)
    else:
        xs, ts, preprocessing_info = sample_multiple_rhos(vec, training_length + 25, n_replicates, obs_noise, variance)

    # Put them together into one library
    collection = []
    for i, (a, b) in enumerate(zip(xs, ts)):
        for j in range(len(a)):
            collection.append(Point(a[j], b[j], "A", i))
    del xs, ts

    # Split train and test set
    ts_train = [point for point in collection if point.time_stamp <= training_length]
    ts_test = [point for point in collection if point.time_stamp > training_length]
    del collection

    # Train model and predict test set
    model = EDM(max_dim=max(int(np.sqrt(training_length)), 10))
    model.train(ts_train)
    _, smap = model.predict(ts_test, hor=max_horizon)

    # Measure performance
    smap = smap.dropna()

    for hor in range(1, max_horizon + 1):
        result_hor = reverse_preprocessing(smap[smap['hor'] == hor], preprocessing_info)
        data = pd.DataFrame(result_hor)
        file_name = f"C:/Users/fleur/Documents/Resultaten/{test}/rho = {rho}/training_length = {training_length}, " \
                    f"n_repl = {n_replicates}, noise = {obs_noise}, var = {variance}, hor = {hor}, " \
                    f"iter = {iter}.csv"
        data.to_csv(file_name, index=False)


def do_multiprocessing(func, argument_list):
    with Pool(16) as pool:
        pool.imap(func=func, iterable=argument_list)
        pool.close()
        pool.join()


def repeatedly_do_EDM(n_iterations=100, n_replicates=1, training_length=25, max_horizon=1, test="begin_conditions", rho=28):

    # Parameters etc.
    np.random.seed(123)
    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    noise_levels = [0.0, 1.0, 2.0, 3.0, 4.0]
    begin_cond_var = [1.0, 4.0, 7.0, 10.0, 13.0]
    rho_var = [1.0, 3.0, 5.0, 7.0, 10.0]

    # ------------------------------------------------------------------------------------------------------------------

    initial_vecs = sample_initial_points(n_iterations)

    if test == "begin_conditions":
        argument_list = list(enumerate(product(noise_levels, begin_cond_var, initial_vecs)))

        if n_replicates == 0:
            argument_list = list(enumerate(product(noise_levels, initial_vecs)))

        partial_function = partial(loop_over_noise_levels,
                                   n_replicates=n_replicates,
                                   training_length=training_length,
                                   max_horizon=max_horizon,
                                   test="begin_conditions",
                                   rho=rho)
    else:
        argument_list = list(enumerate(product(noise_levels, rho_var, initial_vecs)))

        if n_replicates == 0:
            argument_list = list(enumerate(product(noise_levels, initial_vecs)))

        partial_function = partial(loop_over_noise_levels,
                                   n_replicates=n_replicates,
                                   training_length=training_length,
                                   max_horizon=max_horizon,
                                   test="rho",
                                   rho=rho)

    do_multiprocessing(func=partial_function, argument_list=argument_list)

if __name__ == "__main__":

    print("------- rho = 20, begin conditions --------")
    #rho = 20
    count=0
    for test in ["begin_conditions", "rho"]:
        for training_length in [25, 50, 75, 100, 125, 150, 175, 200]:
            repeatedly_do_EDM(n_iterations=100, max_horizon=10, test=test, training_length=training_length, n_replicates=0, rho=20)
            print("Finished round " + str(count) + "of " + str(16))
            count += 1

    # print("------- rho = 28, rho --------")
    # count = 0
    # for test in ["rho"]:
    #     for training_length in [25, 50, 75, 100]:
    #         for n_replicates in [1, 2, 4, 8]:
    #             if count > 10:
    #                 repeatedly_do_EDM(n_iterations=100, max_horizon=10, test=test, training_length=training_length,
    #                                   n_replicates=n_replicates, rho=28)
    #                 print("Finished round " + str(count) + "of " + str(16))
    #             count += 1


    # print("------- rho = 20, begin conditions --------")
    # count = 0
    # for test in ["begin_conditions"]:
    #     for training_length in [25, 50]:
    #         for n_replicates in [1, 2, 4, 8]:
    #
    #             repeatedly_do_EDM(n_iterations=100, max_horizon=10, test=test, training_length=training_length, n_replicates=n_replicates, rho=20)
    #             print("Finished round " + str(count) + "of " + str(16))
    #             count += 1

    # print("------- rho = 20, rho --------")
    # count = 0
    # for test in ["rho"]:
    #     for training_length in [25, 50]:
    #         for n_replicates in [1, 2, 4, 8]:
    #             if count > 2:
    #                 repeatedly_do_EDM(n_iterations=100, max_horizon=10, test=test, training_length=training_length,
    #                                       n_replicates=n_replicates, rho=20)
    #             print("Finished round " + str(count) + " of " + str(16))
    #             count += 1
    #             #0 1 2 done


    # rho = 28
    # count = 0
    # for test in ["begin_conditions", "rho"]:
    #     for training_length in [125, 150, 175, 200]:
    #         for n_replicates in [1, 2, 4, 8, 16]:
    #             repeatedly_do_EDM(n_iterations=100, max_horizon=6, test=test, training_length=training_length,
    #                               n_replicates=n_replicates, rho = rho)
    #             print("Finished round " + str(int(count / (2 * 4 * 5))))
    #
    # rho = 20
    # count = 0
    # for test in ["begin_conditions", "rho"]:
    #     for training_length in [125, 150, 175, 200]:
    #         for n_replicates in [1, 2, 4, 8, 16]:
    #             repeatedly_do_EDM(n_iterations=100, max_horizon=6, test=test, training_length=training_length,
    #                               n_replicates=n_replicates, rho = rho)
    #             print("Finished round " + str(int(count / (2 * 4 * 5))))