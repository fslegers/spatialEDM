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
        x, _ = preprocessing(x, t, loc=i)
        x += np.random.normal(0.0, obs_noise)
        x, _ = preprocessing(x, t, loc=i)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t


def sample_multiple_rhos(vec_0, n_points, n_repl, obs_noise, var, rho):

    list_x, list_t = [], []

    i = 0
    while i < n_repl:
        rho_ = np.random.normal(rho, var)

        x, t = sample_lorenz(vec_0, [10, rho_, 8/3], n_points, 0.0, n_points)
        x, _ = preprocessing(x, t, loc=i+1)
        x += np.random.normal(0.0, obs_noise, size=x.shape)
        x, _ = preprocessing(x, t, loc=i+1)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t


def perform_EDM(og, initial_point, length, n_replicates, noise, variance, max_dim=10, test="begin_conditions", rho=28):

    x, t = og[0], og[1]
    original_length = len(x) - 5

    if n_replicates > 0:
        if test == "begin_conditions":
            new_x, new_t = sample_multiple_initial_values(initial_point, length, n_replicates,
                                                                             noise, variance, rho)
        else:
            new_x, new_t = sample_multiple_rhos(initial_point, length, n_replicates,
                                                                   noise, variance, rho)

        # from original time series, select last values and add these to the list of replicates
        xs, ts = [x[-(length+5):]] + new_x, [t[-(length+5):]] + new_t

    else:
        if len(x) != length:
            xs, ts = [x[-(length+5):]], [t[-(length+5):]]
        else:
            xs, ts = x, t

    # Put them together into one library
    collection = []
    for i, (a, b) in enumerate(zip(xs, ts)):
        for j in range(len(a)):
            collection.append(Point(a[j], b[j], "A", i))
    del(xs, ts, x, t)

    # Split train and test set
    ts_train = [point for point in collection if point.time_stamp < original_length]
    ts_test = [point for point in collection if point.time_stamp >= original_length]
    del(collection)

    # Train model and predict test set
    model = EDM()
    model.train(ts_train, max_dim=max_dim)
    _, smap = model.predict(ts_test, hor=1)

    # Measure performance
    smap = smap.dropna()
    results = calculate_performance(smap)

    return results


# def calculate_boxplot_values(data):
#
#     data_without_none = []
#     none_count = 0
#
#     for entry in data:
#         if entry is None:
#             none_count += 1
#
#         else:
#             data_without_none.append(entry)
#
#     # Sort the data
#     sorted_data = sorted(data_without_none)
#
#     # Calculate quartiles
#     q1 = np.percentile(sorted_data, 25)
#     median = np.percentile(sorted_data, 50)
#     q3 = np.percentile(sorted_data, 75)
#
#     # Calculate IQR
#     iqr = q3 - q1
#
#     # Calculate whisker values
#     lower_whisker = q1 - 1.5 * iqr
#     upper_whisker = q3 + 1.5 * iqr
#
#     # Find outliers
#     outliers = [x for x in sorted_data if x < lower_whisker or x > upper_whisker]
#
#     # Calculate min and max
#     min_val = min(sorted_data)
#     max_val = max(sorted_data)
#
#     return {
#         'min': min_val,
#         'max': max_val,
#         'median': median,
#         'q1': q1,
#         'q3': q3,
#         'iqr': iqr,
#         'lower_whisker': lower_whisker,
#         'upper_whisker': upper_whisker,
#         'outliers': outliers,
#         'none_count': none_count
#     }


def process_pair(pair, initial_points, original_length, replicates_length, test="begin_conditions", rho=28):

    noise, variance = pair
    list_of_n_replicates = []

    # Determine max dimensions
    E_og = int(np.sqrt(original_length))
    E_new = int(np.sqrt(replicates_length))

    for i in range(len(initial_points)):

        v0 = initial_points[i]

        # Sample original trajectory
        x, t = sample_lorenz(v0, [10, rho, 8/3], original_length + 9, 0.0)
        x, _ = preprocessing(x, t, loc=0)
        x += np.random.normal(0.0, noise, size=x.shape)
        x, _ = preprocessing(x, t, loc=0)
        og_ts = [x, t]

        # Perform EDM with on original time series
        RMSE_full_ts = perform_EDM(og_ts, v0, original_length, 0, noise, 0, E_og, test, rho)['RMSE']

        # Repeat for last part of original trajectory
        # and add replicates
        for n_replicates in range(0,17):
            RMSE_replicates = perform_EDM(og_ts, v0, replicates_length, n_replicates, noise, variance, E_new, test, rho)['RMSE']

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


def loop(original_length, replicates_length, rho=28):

    n_iterations = 100
    noise_levels = [0.0, 0.05, 0.1, 0.25]
    begin_cond_var = [1.0, 5.0, 10.0]
    rho_var = [1.0, 5.0, 10.0]
    n_processes = 12

    # Sample initial point for each iteration
    initial_points = sample_initial_points(n_iterations, rho)

    # Define partial function
    partial_function_begin_cond = partial(process_pair, initial_points=initial_points, original_length=original_length,
                                          replicates_length=replicates_length, test='begin_conditions', rho=rho)

    partial_function_rho = partial(process_pair, initial_points=initial_points, original_length=original_length,
                                   replicates_length=replicates_length, test='rho', rho=rho)

    # First, test with variance in begin conditions
    argument_list = list(product(noise_levels, begin_cond_var))
    result_list = run_imap_multiprocessing(func=partial_function_begin_cond,
                                           argument_list=argument_list,
                                           num_processes=n_processes)
    data = pd.DataFrame(result_list)
    file_name = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 5/begin conditions/{original_length} vs {replicates_length}, rho = {rho}.csv"
    data.to_csv(file_name, index=False)
    del(argument_list, result_list, data)

    # Then, test with variance in parameter rho
    argument_list = list(product(noise_levels, rho_var))
    result_list = run_imap_multiprocessing(func=partial_function_rho,
                                           argument_list=argument_list,
                                           num_processes=n_processes)
    data = pd.DataFrame(result_list)
    file_name = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/length vs replicates/test set = 5/rho/{original_length} vs {replicates_length}, rho = {rho}.csv"
    data.to_csv(file_name, index=False)
    del (argument_list, result_list, data)


if __name__ == '__main__':
    for rho in [28, 20]:
        print("--------- rho = {rho} ---------".format(rho=rho))

        print("Starting experiment with 96 vs 48")
        original_length = 96
        replicates_length = 48
        loop(original_length, replicates_length, rho)

        print("Starting experiment with 48 vs 24")
        original_length = 48
        replicates_length = 24
        loop(original_length, replicates_length, rho)

        print("Starting experiment with 24 vs 12")
        original_length = 24
        replicates_length = 12
        loop(original_length, replicates_length, rho)




        
