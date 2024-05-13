from multiprocessing import Pool
from functools import partial

from src.classes import *
from src.simulate_lorenz import *


def sample_lorenz(vec_0, params, n_points, obs_noise, offset=0):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i + offset for i in range(len(t))]
    return x, t

def calculate_performance(result):

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


def perform_EDM(index, length, n_replicates, noise, variance, x, t):
    # Determine max dimensions
    max_dim = int(np.sqrt(length))

    xs, ts = [], []

    # Get original trajectory
    x_, t_ = x[index:index + length + 10], [time for time in np.arange(0, length + 10)]

    x_, _ = preprocessing(x_, t_, loc=0)
    x_ = x_ + np.random.normal(0, noise, size=len(x_))
    x_, _ = preprocessing(x_, t_, loc=0)

    xs.append(x_)
    ts.append(t_)

    # Add replicates
    if variance == 0.0:
        repl_indices = [index for _ in range(n_replicates)]
    else:
        repl_indices = np.random.randint(index - variance, index + variance, size=n_replicates)

    for j in repl_indices:
        x_, t_ = x[j:j+length], [time for time in np.arange(0, length)]
        x_, _ = preprocessing(x_, t_, loc=j)
        x_ += np.random.normal(0, noise, size=len(x_))
        x_, _ = preprocessing(x_, t_, loc=j)
        xs.append(x_)
        ts.append(t_)

    # Put them together into one library
    collection = []
    for index, (a, b) in enumerate(zip(xs, ts)):
        for j in range(len(a)):
            collection.append(Point(a[j], b[j], "", index))
    del (xs, ts, x, t, x_, t_)

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
    results = calculate_performance(smap)

    return results


def partial_function(variance, indices, n_replicates, length, noise, rho, x, t):
    results = []

    for i in indices:
        partial_result = perform_EDM(i, length, n_replicates, noise, variance, x, t)
        RMSE = partial_result['RMSE']

        row = {'noise': noise, 'variance': variance, 'rho': rho, 'length': length, 'n_repl': n_replicates,
               'RMSE': RMSE}

        results.append(row)

    return pd.DataFrame(results)


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)
    result_list = []
    for result in pool.imap(func=func, iterable=argument_list):
        result_list.append(result)
    combined_df = pd.concat(result_list, ignore_index=True)
    return combined_df


def loop(rho, n_repl, length, noise):
    np.random.seed(123)

    n_processes = 8
    n_iterations = 250
    variances = np.arange(0, 100, 5)

    # Sample initial point for each iteration
    indices = np.random.randint(100, 900-length, size=n_iterations)

    x, _, _, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 8000, 140, 0)
    x, t = sample_from_ts(x[1000:], t[1000:], sampling_interval=5, n_points=1000)
    t = [i for i in range(len(t))]

    # Define partial function
    partial_func = partial(partial_function,
                           indices=indices,
                           n_replicates=n_repl,
                           length=length,
                           noise=noise,
                           rho=rho,
                           x=x,
                           t=t)

    # First, test with variance in begin conditions
    result_list = run_imap_multiprocessing(func=partial_func,
                                           num_processes=n_processes,
                                           argument_list=variances)

    data = pd.DataFrame(result_list)
    file_name = (f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/"
                 f"RMSE vs variance among replicates/test_2/rho = {rho}/"
                 f"length = {length}, noise = {noise}, n_repl = {n_repl}.csv")
    data.to_csv(file_name, index=False)
    del (result_list, data)


if __name__ == "__main__":

    rho = 28
    for n_repl in [2, 4, 8, 16]:
        for length in [25, 50, 75]:
            for noise in [0.0, 2.0, 4.0]:
                loop(rho, n_repl, length, noise)


