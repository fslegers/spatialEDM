from src.classes import *
from src.simulate_lorenz import *
import pandas as pd
import warnings
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

def sample_lorenz(vec_0, params, n_points, obs_noise):

    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise) #TODO: maybe tmax should be different?
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i for i in range(len(t))]

    return x, t

def sample_multiple_initial_values(n_replicates, vec_0, std_dev, params, n_points, obs_noise):
    list_x, list_t = [], []
    for i in range(n_replicates):
        x_0 = np.random.normal(vec_0[0], std_dev)
        y_0 = np.random.normal(vec_0[1], std_dev)
        z_0 = np.random.normal(vec_0[2], std_dev)
        x, t = sample_lorenz([x_0, y_0, z_0], params, n_points, obs_noise)
        list_x.append(x)
        list_t.append(t)
    return list_x, list_t

def process_iteration(args):
    init_var, n_replicates, vec_0, ts_length, obs_noise, horizon = args
    results = []

    xs, ts = sample_multiple_initial_values(n_replicates, vec_0, init_var, [10, 28, 8 / 3], ts_length + 25, obs_noise)

    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    ts_train = [point for point in collection if point.time_stamp <= ts_length]
    ts_test = [point for point in collection if point.time_stamp > ts_length]

    model = EDM()
    model.train(ts_train, max_dim=10)

    simplex, smap = model.predict(ts_test, hor=horizon)

    for i in range(len(simplex)):
        row_simplex = {'obs_noise': obs_noise, 'init_var': init_var, 'hor': simplex['hor'][i], 'obs': simplex['obs'][i],
                       'pred_simplex': simplex['pred'][i]}
        row_smap = {'obs_noise': obs_noise, 'init_var': init_var, 'hor': smap['hor'][i], 'obs': smap['obs'][i],
                    'pred_smap': smap['pred'][i]}
        row = {**row_simplex, **row_smap}
        results.append(row)

    return results

if __name__ == "__main__":

    n_replicates = 12
    n_iterations = 100
    training_length = 25
    max_horizon = 8

    np.random.seed(123)
    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8/3], 2000, 30, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    initial_vecs = []
    indices = np.random.randint(len(x), size=n_iterations)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    initial_point_variances = [0, 1.0, 2.0, 3.0, 4.0, 5,0, 6.0, 7.0, 8.0, 9.0]
    obs_noises = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    for obs_noise_index in range(len(obs_noises)):
        obs_noise = obs_noises[obs_noise_index]

        print("\n Starting round " + str(obs_noise_index + 1) + " of " + str(len(obs_noises)))

        total_results = pd.DataFrame(columns=['obs_noise', 'init_var', 'hor', 'obs', 'pred_simplex', 'pred_smap'])
        for vec_0 in tqdm(initial_vecs, desc="Processing", unit="iteration"):

            with ProcessPoolExecutor(max_workers=10) as executor:
                args = [(init_var, n_replicates, vec_0, training_length, obs_noise, max_horizon) for init_var in initial_point_variances]
                results_list = list(tqdm(executor.map(process_iteration, args), total=len(args), desc="Processing Initial Point Variance"))

            results_df = pd.concat([pd.DataFrame(result) for result in results_list], ignore_index=True)
            total_results = pd.concat([total_results, results_df])

            path_name = "./initial point variance/obs_noise = " + str(obs_noise) + ", n_iterations = " + str(n_iterations) + ", n_replicates = " + str(n_replicates) + ", ts_length = " + str(
                training_length) + ", hor = " + str(max_horizon) + ".csv"
            total_results.to_csv(path_name, index=False)



