from itertools import product
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

from src.classes import *
from src.simulate_lorenz import *

def sample_lorenz(vec_0, params, n_points, obs_noise, offset=0):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i + offset for i in range(len(t))]
    return x, t


def calculate_performance(result):

    # # Reverse preprocessing
    # result = reverse_preprocessing(result, preprocessing_info)

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
    indices = np.random.randint(len(x), size=n_points)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    return initial_vecs


def sample_multiple_initial_values(vec_0, n_points, n_repl, obs_noise, var, rho):

    list_x, list_t = [], []

    i = 0
    while i < n_repl:
        x_0 = np.random.normal(vec_0[0], var)
        y_0 = np.random.normal(vec_0[1], var)
        z_0 = np.random.normal(vec_0[2], var)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, rho, 8/3], n_points, 0.0, n_points)

        # Preprocessing
        x, _ = preprocessing(x, t, loc=i)
        x += np.random.normal(0.0, obs_noise, size=len(x))
        x, _ = preprocessing(x, t, loc=i)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t


def perform_EDM_og(full_x, index, length, noise, max_dim=10):

    # Sample original trajectory
    x = full_x[index : index + length + 5]
    t = [j for j in range(len(x))]

    # Add noise + preprocessing
    x, _ = preprocessing(x, t, loc=0)
    x += np.random.normal(0, noise, size=len(x))
    x, _ = preprocessing(x, t, loc=0)

    # Turn into library
    collection = []
    for j in range(len(x)):
        collection.append(Point(x[j], t[j], "", 0))
    del(x, t)

    # Split train and test set
    ts_train = [point for point in collection if point.time_stamp < length]
    ts_test = [point for point in collection if point.time_stamp >= length]
    del(collection)

    # Train model and predict test set
    model = EDM()
    model.train(ts_train, max_dim=max_dim)
    _, smap = model.predict(ts_test, hor=1)

    # Measure performance
    smap = smap.dropna()
    results = calculate_performance(smap)

    return results


def perform_EDM_repl(full_x, index, length, n_repl, noise, var, max_dim=10):

    collection = []

    for i in range(n_repl + 1):

        # Sample original trajectory
        if i == 0:
            x = full_x[index : index + length + 5]
        else:
            index_repl = np.random.randint(low=index-var, high=index+var)
            x = full_x[index_repl : index_repl + length]

        t = [j for j in range(len(x))]

        x, _ = preprocessing(x, t, loc=i)
        x += np.random.normal(0, noise, size=len(x))
        x, _ = preprocessing(x, t, loc=i)

        # Turn into library
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "", i))

    # Split train and test set
    ts_train = [point for point in collection if point.time_stamp < length]
    ts_test = [point for point in collection if point.time_stamp >= length]

    if len(ts_test) < 5:
        print("ERROR!!!")

    del(collection)

    # Train model and predict test set
    model = EDM()
    try:
        model.train(ts_train, max_dim=max_dim)
    except:
        print(max_dim)
    _, smap = model.predict(ts_test, hor=1)

    # Measure performance
    smap = smap.dropna()

    try:
        results = calculate_performance(smap)
    except:
        print('Error calculating performance')

    return results


def process_pair_b(pair, initial_points, x, original_length, replicates_length):

    noise, variance = pair
    list_of_n_replicates = []

    # Determine max dimensions
    E_og = int(np.sqrt(original_length))
    E_new = int(np.sqrt(replicates_length))

    for index in initial_points:

        # Perform EDM with on original time series
        RMSE_full_ts = perform_EDM_og(x, index, original_length, noise, E_og)['RMSE']

        # Repeat for last half of original trajectory and add replicates
        for n_replicates in range(0,17):
            index_new = index + replicates_length
            RMSE_replicates = perform_EDM_repl(x, index_new, replicates_length, n_replicates, noise, variance, E_new)['RMSE']

            if RMSE_replicates <= RMSE_full_ts:
                list_of_n_replicates.append(n_replicates)
                break

            if n_replicates == 16 and RMSE_replicates > RMSE_full_ts:
                list_of_n_replicates.append(18)

    row = {'noise': noise, 'variance': variance, 'n_replicates': list_of_n_replicates}

    return row


def run_imap_multiprocessing(func, argument_list, num_processes):
    pool = Pool(processes=num_processes)
    result_list = []
    for result in pool.imap(func=func, iterable=argument_list):
        result_list.append(result)
    return result_list


def loop_b(original_length, replicates_length, rho=28):

    n_iterations = 100
    noise_levels = [0.0, 0.05, 0.1, 0.25]
    t_repl = [1.0, 5.0, 10.0]
    n_processes = 8

    # simulate one Lorenz attractor without transient
    x, _, _, t = simulate_lorenz([1, 1, 1], [10, rho, 8 / 3], 8000, 140, 0)
    x, t = sample_from_ts(x[1000:], t[1000:], sampling_interval=5, n_points=1000)
    del(t)

    # select different initial points from this trajectory
    indices = np.random.randint(low=200, high=800, size=n_iterations)

    # Define partial function
    partial_function = partial(process_pair_b,
                               x=x,
                               initial_points=indices,
                               original_length=original_length,
                               replicates_length=replicates_length)

    # First, test with variance in begin conditions
    argument_list = list(product(noise_levels, t_repl))
    result_list = run_imap_multiprocessing(func=partial_function,
                                           argument_list=argument_list,
                                           num_processes=n_processes)
    data = pd.DataFrame(result_list)
    file_name = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 5/b/{original_length} vs {replicates_length}, rho = {rho}.csv"
    data.to_csv(file_name, index=False)
    del(argument_list, result_list, data)


if __name__ == '__main__':

    test = 'b'
    for rho in [28, 20]:
        print("--------- rho = {rho} ---------".format(rho=rho))

        print("Starting experiment with 24 vs 12")
        original_length = 24
        replicates_length = 12
        loop_b(original_length, replicates_length, rho)

        print("Starting experiment with 48 vs 24")
        original_length = 48
        replicates_length = 24
        loop_b(original_length, replicates_length, rho)

        print("Starting experiment with 96 vs 48")
        original_length = 96
        replicates_length = 48
        loop_b(original_length, replicates_length, rho)

        
