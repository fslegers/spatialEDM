import time
import matplotlib.pyplot as plt
from src.classes import *
from src.simulate_lorenz import *
import pandas as pd
import warnings
import os

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


if __name__ == "__main__":

    n_replicates = 1
    n_iterations = 5
    ts_length = 25
    horizon = 1
    test = ["initial_point"]

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

    results_initial_point_variance = pd.DataFrame(columns=['obs_noise', 'init_var', 'hor', 'obs', 'pred_simplex', 'pred_smap'])
    results_rho_variance = pd.DataFrame(columns=['obs_noise', 'param_var', 'hor', 'obs', 'pred_simplex', 'pred_smap'])

    # For each initial point
    for vec_0 in initial_vecs:

        print("new initial point")

        # Test different levels of observational noise
        for obs_noise in [0, 1.0, 2.0]:

            # Test different levels of variance initial values
            if("initial_point" in test):
                for init_var in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:

                    # Sample multiple time series
                    xs, ts = sample_multiple_initial_values(n_replicates, vec_0, init_var, [10, 28, 8/3], ts_length, obs_noise)

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

                    for i in range(len(simplex)):
                        row_simplex = {'obs_noise': obs_noise, 'init_var': init_var, 'hor': simplex['hor'][i], 'obs': simplex['obs'][i], 'pred_simplex': simplex['pred'][i]}
                        row_smap = {'obs_noise': obs_noise, 'init_var': init_var, 'hor': smap['hor'][i], 'obs': smap['obs'][i], 'pred_smap': smap['pred'][i]}

                        row = {**row_simplex, **row_smap}

                        results_initial_point_variance = pd.concat([results_initial_point_variance, pd.DataFrame([row])], ignore_index=True)



            # Test different levels of parameter variance
            if ("rho" in test):
                for param_var in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:

                    # Sample multiple time series
                    xs, ts = sample_multiple_rhos(n_replicates, vec_0, [10, 28, 8/3], param_var, ts_length, obs_noise)

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
                    simplex, smap = model.predict(ts_test, hor=3)

                    if len(simplex) != len(smap):
                        print('length simplex and smap is not equal. \n'
                              'This could be problematic for merging the two dictionaries.')

                    for i in range(len(simplex)):
                        row_simplex = {'obs_noise': obs_noise, 'param_var': init_var, 'hor': simplex['hor'][i], 'obs': simplex['obs'][i], 'pred_simplex': simplex['pred'][i]}
                        row_smap = {'obs_noise': obs_noise, 'param_var': init_var, 'hor': smap['hor'][i], 'obs': smap['obs'][i], 'pred_smap': smap['pred'][i]}

                        row = {**row_simplex, **row_smap}

                        results_rho_variance = pd.concat([results_rho_variance, pd.DataFrame([row])], ignore_index=True)

            if "init_point" in test:
                path_name = "initial point variance, n_iterations = " + str(n_iterations) + ", n_replicates = " + str(
                    n_replicates) + ", ts_length = " + str(ts_length) + ", hor = " + str(horizon) + ".csv"
                results_initial_point_variance.to_csv(path_name, index=False)

            if "rho" in test:
                path_name = "rho variance, n_iterations = " + str(n_iterations) + ", n_replicates = " + str(n_replicates) + ", ts_length = " + str(
                    ts_length) + ", hor = " + str(horizon) + ".csv"
                results_rho_variance.to_csv(path_name, index=False)




