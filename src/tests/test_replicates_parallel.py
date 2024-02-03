from src.classes import *
from src.simulate_lorenz import *
import pandas as pd
import warnings
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

def sample_lorenz(vec_0, params, n_points, obs_noise):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points, n_points, obs_noise) #TODO: maybe tmax should be different?
    t = [int(i) for i in t]
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

def sample_multiple_rhos(n_replicates, vec_0, params, std_dev, n_points, obs_noise):
    list_x, list_t = [], []
    for i in range(n_replicates):
        rho = np.random.normal(params[1], std_dev)
        x, t = sample_lorenz(vec_0, [params[0], rho, params[2]], n_points, obs_noise)
        list_x.append(x)
        list_t.append(t)
    return list_x, list_t


def process_iteration_initial_values(init_var, n_replicates, vec_0, ts_length, obs_noise, horizon):

    xs, ts = sample_multiple_initial_values(n_replicates, vec_0, init_var, [10, 28, 8 / 3], ts_length, obs_noise)

    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    ts_train = [point for point in collection if point.time_stamp <= 35]
    ts_test = [point for point in collection if point.time_stamp > 35]

    model = EDM()
    model.train(ts_train, max_dim=10)

    simplex, smap = model.predict(ts_test, hor=horizon)

    results = []
    for i in range(len(simplex)):
        row_simplex = {'obs_noise': obs_noise, 'init_var': init_var, 'hor': simplex['hor'][i], 'obs': simplex['obs'][i],
                       'pred_simplex': simplex['pred'][i]}
        row_smap = {'obs_noise': obs_noise, 'init_var': init_var, 'hor': smap['hor'][i], 'obs': smap['obs'][i],
                    'pred_smap': smap['pred'][i]}
        row = {**row_simplex, **row_smap}
        results.append(row)

    return results


def process_iteration_rhos(param_var, n_replicates, vec_0, ts_length, obs_noise, horizon):

    # Sample multiple time series
    xs, ts = sample_multiple_rhos(n_replicates, vec_0, [10, 28, 8 / 3], param_var, ts_length, obs_noise)

    # Concatenate time series
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    # Split time series
    ts_train = [point for point in collection if point.time_stamp <= 35]
    ts_test = [point for point in collection if point.time_stamp > 35]

    # Train model
    model = EDM()
    model.train(ts_train, max_dim=10)

    # Predict test points
    simplex, smap = model.predict(ts_test, hor=horizon)

    results = []
    for i in range(len(simplex)):
        row_simplex = {'obs_noise': obs_noise, 'init_var': param_var, 'hor': simplex['hor'][i], 'obs': simplex['obs'][i],
                       'pred_simplex': simplex['pred'][i]}
        row_smap = {'obs_noise': obs_noise, 'init_var': param_var, 'hor': smap['hor'][i], 'obs': smap['obs'][i],
                    'pred_smap': smap['pred'][i]}
        row = {**row_simplex, **row_smap}
        results.append(row)

    return results


if __name__ == "__main__":

    n_replicates = 3
    n_iterations = 3
    ts_length = 75
    horizon = 2
    run_tests = ["rho"]

    # Set seed and navigate to results directory
    np.random.seed(123)
    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8/3], 2000, 30, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select a number (50) of initial points from this giant trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=n_iterations)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # initial_point_variances = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
    # rho_variances = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    initial_point_variances = [0, 0.5, 5.0]
    rho_variances = [0, 0.5, 5.0]

    obs_noises = [0.0, 1.0, 2.0]
    for obs_noise_index in range(len(obs_noises)):
        obs_noise = obs_noises[obs_noise_index]

        print("\n Starting round " + str(obs_noise_index + 1) + " of " + str(len(obs_noises)))

        results_initial_point_variance = pd.DataFrame(columns=['obs_noise', 'init_var', 'hor', 'obs', 'pred_simplex', 'pred_smap'])
        results_rho_variance = pd.DataFrame(columns=['obs_noise', 'param_var', 'hor', 'obs', 'pred_simplex', 'pred_smap'])

        for vec_0 in tqdm(initial_vecs, desc="Processing", unit="iteration"):

            # Test different levels of variance initial values
            if("initial_point" in run_tests):
                with ProcessPoolExecutor(max_workers=10) as executor:
                    results_list = list(executor.map(process_iteration_initial_values, initial_point_variances, [n_replicates] * len(initial_point_variances),
                                                     [vec_0] * len(initial_point_variances),
                                                     [ts_length] * len(initial_point_variances), [obs_noise] * len(initial_point_variances),
                                                     [horizon] * len(initial_point_variances)))

                # Concatenate results into a single list
                results_df = pd.concat([pd.DataFrame(result) for result in results_list], ignore_index=True)
                results_initial_point_variance = pd.concat([results_initial_point_variance, results_df])

                path_name = "./initial point variance/obs_noise = " + str(obs_noise) + ", n_iterations = " + str(n_iterations) + ", n_replicates = " + str(
                    n_replicates) + ", ts_length = " + str(ts_length) + ", hor = " + str(horizon) + ".csv"
                results_initial_point_variance.to_csv(path_name, index=False)


            # Test different levels of parameter variance
            if ("rho" in run_tests):
                with ProcessPoolExecutor(max_workers=10) as executor:
                    results_list = list(executor.map(process_iteration_rhos, rho_variances,
                                                     [n_replicates] * len(rho_variances),
                                                     [vec_0] * len(rho_variances),
                                                     [ts_length] * len(rho_variances),
                                                     [obs_noise] * len(rho_variances),
                                                     [horizon] * len(rho_variances)))

                # Concatenate results into a single list
                results_df = pd.concat([pd.DataFrame(result) for result in results_list], ignore_index=True)
                results_rho_variance = pd.concat([results_rho_variance, results_df])

                path_name = "./rho variance/obs_noise = " + str(obs_noise) + ", n_iterations = " + str(n_iterations) + ", n_replicates = " + str(n_replicates) + ", ts_length = " + str(
                    ts_length) + ", hor = " + str(horizon) + ".csv"
                results_rho_variance.to_csv(path_name, index=False)







