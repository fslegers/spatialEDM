from multiprocessing import Pool
from functools import partial

import pandas as pd

from src.classes import *
from src.simulate_lorenz import *


def sample_lorenz(vec_0, params, n_points, obs_noise, offset=0):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i + offset for i in range(len(t))]
    return x, t


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


def sample_initial_points(n_points, rho):
    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(x.shape[0], size=n_points)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    return initial_vecs


def sample_multiple_initial_values(vec_0, n_points, n_repl, obs_noise, var, rho):
    list_x, list_t, list_preprocessing = [], [], []

    i = 0
    while i < n_repl:
        x_0 = np.random.normal(vec_0[0], var)
        y_0 = np.random.normal(vec_0[1], var)
        z_0 = np.random.normal(vec_0[2], var)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, rho, 8 / 3], n_points, obs_noise, n_points)

        # Preprocessing
        x, preprocessing_info = preprocessing(x, t, loc=i)
        list_preprocessing.append(preprocessing_info)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t, list_preprocessing


def sample_multiple_rhos(vec_0, n_points, n_repl, obs_noise, var, rho):
    list_x, list_t, list_preprocessing = [], [], []

    i = 0
    while i < n_repl:
        rho = np.random.normal(28, var)
        x, t = sample_lorenz(vec_0, [10, rho, 8 / 3], n_points, obs_noise, n_points)

        # Preprocessing
        x, preprocessing_info = preprocessing(x, t, loc=i + 1)
        list_preprocessing.append(preprocessing_info)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t, list_preprocessing

def perform_EDM(ts, initial_point, length, n_replicates, noise, variance, test, rho):

    # Determine max dimensions
    max_dim = int(np.sqrt(length))

    # Get original time series
    x, t, preprocessing_info = ts[0], ts[1], ts[2]

    # Create replicate time series
    if test == "begin_conditions":
        new_x, new_t, _ = sample_multiple_initial_values(initial_point, length, n_replicates, noise, variance, rho)
    else:
        new_x, new_t, _ = sample_multiple_rhos(initial_point, length, n_replicates, noise, variance, rho)

    # from original time series, select last values and add these to the list of replicates
    xs, ts = [x[-(length + 9):]] + new_x, [t[-(length + 9):]] + new_t

    # Put them together into one library
    collection = []
    for i, (a, b) in enumerate(zip(xs, ts)):
        for j in range(len(a)):
            collection.append(Point(a[j], b[j], "", i))
    del (xs, ts, x, t)

    # Split train and test set
    ts_train = [point for point in collection if point.time_stamp < length]
    ts_test = [point for point in collection if point.time_stamp >= length]
    del (collection)

    # Train model and predict test set
    model = EDM()
    model.train(ts_train, max_dim=max_dim)
    _, smap = model.predict(ts_test, hor=1)

    # Measure performance
    smap = smap.dropna(how='any')
    results = calculate_performance(smap, [preprocessing_info])
    results['dim'] = model.dim
    results['theta'] = model.theta
    results['rmse_list_theta'] = model.results_smap['rmse_list']
    results['rmse_list_dim'] = model.results_simplex['rmse_list']

    return results

def partial_function(variance, initial_points, n_replicates, length, noise, test, rho):
    results = []

    for i in range(len(initial_points)):
        v0 = initial_points[i]

        # Sample original trajectory
        x, t = sample_lorenz(v0, [10, rho, 8 / 3], length + 9, noise)
        x, preprocessing_info = preprocessing(x, t, loc=0)
        ts = [x, t, preprocessing_info]

        # Perform EDM with on original time series
        partial_result = perform_EDM(ts, v0, length, n_replicates, noise, variance, test, rho)
        RMSE, corr, dim, theta = partial_result['RMSE'], partial_result['corr'], partial_result['dim'], partial_result['theta']
        RMSE_list_theta, RMSE_list_dim = partial_result['rmse_list_theta'], partial_result['rmse_list_dim']

        row = {'noise': noise, 'variance': variance, 'rho': rho, 'length': length, 'test': test, 'n_repl': n_replicates,
               'RMSE': RMSE, 'corr': corr, 'dim': dim, 'theta': theta, 'RMSE_list_theta': RMSE_list_theta, 'RMSE_list_dim':RMSE_list_dim}

        results.append(row)

    return pd.DataFrame(results)


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)
    result_list = []
    for result in pool.imap(func=func, iterable=argument_list):
        result_list.append(result)
    combined_df = pd.concat(result_list, ignore_index=True)
    return combined_df


def loop(rho, noise, length, test, n_replicates):
    np.random.seed(123)

    n_processes = 8
    n_iterations = 100
    variances = np.arange(0.0, 12.0, 3.0)

    # Sample initial point for each iteration
    initial_points = sample_initial_points(n_iterations, rho)

    # Define partial function
    partial_func = partial(partial_function,
                           initial_points=initial_points,
                           n_replicates=n_replicates,
                           length=length,
                           noise=noise,
                           test=test,
                           rho=rho)

    # First, test with variance in begin conditions
    result_list = run_imap_multiprocessing(func=partial_func,
                                           num_processes=n_processes,
                                           argument_list=variances)
    data = pd.DataFrame(result_list)
    file_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
                 f"hyperparameters for replicates/rho = {rho}/{test}/"
                 f"length = {length}, noise = {noise}, n_repl = {n_replicates}.csv")
    data.to_csv(file_name, index=False)
    del (result_list, data)


if __name__ == '__main__':

    rho = 28
    test = "rho"
    n_replicates = 8

    print(f"Testing {test} for rho = {rho}")
    for noise in [0.0, 2.0, 4.0]:
        for len in [25, 50, 75, 100]:
            loop(rho, noise, len, test, n_replicates)

    ####################################################
    test = "begin_conditions"

    print(f"Testing {test} for rho = {rho}")
    for noise in [0.0, 2.0, 4.0]:
        for len in [25, 50, 75, 100]:
            loop(rho, noise, len, test, n_replicates)

    ####################################################

    rho = 20
    test = "rho"

    print(f"Testing {test} for rho = {rho}")
    for noise in [0.0, 2.0, 4.0]:
        for len in [25, 50, 75, 100]:
            loop(rho, noise, len, test, n_replicates)

    ####################################################
    test = "begin_conditions"

    print(f"Testing {test} for rho = {rho}")
    for noise in [0.0, 2.0, 4.0]:
        for len in [25, 50, 75, 100]:
            loop(rho, noise, len, test, n_replicates)







