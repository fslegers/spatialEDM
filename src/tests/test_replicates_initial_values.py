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
        x, preprocessing_info = preprocessing(x, t, loc=i+1)
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



def loop_over_noise_levels(argument, initial_vecs, n_replicates, training_length, max_horizon, test="begin_conditions", rho):

    obs_noise = argument[0]
    variance = argument[1]

    for vec in initial_vecs:

        # Sample replicate time series
        if test == "begin_conditions":
            xs, ts, preprocessing_info = sample_multiple_initial_values(vec, training_length + 10, n_replicates, obs_noise, variance)
        else:
            xs, ts, preprocessing_info = sample_multiple_rhos(vec, training_length + 10, n_replicates, obs_noise, variance)

        # Put them together into one library
        collection = []
        for i, (a, b) in enumerate(zip(xs, ts)):
                for j in range(len(a)):
                    collection.append(Point(a[j], b[j], "A", i))
        del (xs, ts)

        # Split train and test set
        ts_train = [point for point in collection if point.time_stamp < training_length]
        ts_test = [point for point in collection if point.time_stamp >= training_length]
        del (collection)

        # Train model and predict test set
        model = EDM(horizon = max_horizon, max_dim=int(np.sqrt(training_length)))
        model.train(ts_train)
        _, smap = model.predict(ts_test, hor=max_horizon)

        # Measure performance
        smap = smap.dropna()

        result = []
        for hor in range(1, max_horizon + 1):
            result_hor = reverse_preprocessing(smap[smap['hor'] == hor], preprocessing_info)
            result += result_hor

        # Save results
        data = pd.DataFrame(result)
        file_name = f"C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/{test}/rho = {rho}/training_length = {training_length}, noise = {obs_noise}, var = {variance}.csv"
        data.to_csv(file_name, index=False)
        del (smap, result)


def do_multiprocessing(func, argument_list):
    with Pool(8) as pool:
        pool.imap(func=func, iterable=argument_list)

def repeatedly_do_EDM(n_iterations=50, n_replicates=0, training_length=25, max_horizon=1, test = "begin_conditions"):

    # Parameters etc.
    np.random.seed(123)
    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    noise_levels = [0.0, 1.0, 2.0, 3.0]
    begin_cond_var = [1.0, 4.0, 7.0, 10.0]
    rho_var = [1.0, 3.0, 5.0, 7.0]
    # ------------------------------------------------------------------------------------------------------------------

    initial_vecs = sample_initial_points(n_iterations)

    if test == "begin_conditions":
        argument_list = product(noise_levels, begin_cond_var)
        partial_function = partial(loop_over_noise_levels, initial_vecs, n_replicates, training_length, max_horizon, "begin_conditions")
    else:
        argument_list = product(noise_levels, rho_var)
        partial_function = partial(loop_over_noise_levels, initial_vecs, n_replicates, training_length, max_horizon, "rho")

    do_multiprocessing(func=partial_function, argument_list=argument_list)






