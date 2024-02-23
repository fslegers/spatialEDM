import pandas as pd

from src.classes import *
from src.simulate_lorenz import *
import numpy as np


def calculate_performance(result):
    diff = np.subtract(result['obs'], result['pred'])

    performance = {}

    performance['MAE'] = np.mean(abs(diff))
    performance['RMSE'] = math.sqrt(np.mean(np.square(diff)))

    try:
        performance['corr'] = pearsonr(result['obs'], result['pred'])[0]
    except:
        performance['corr'] = None

    return performance


def sample_lorenz(vec_0, params, n_points, obs_noise):
    x, y, z, t = simulate_lorenz(vec_0, params, n_points, n_points, obs_noise)
    t = [int(i) for i in t]
    return x, t


def test_with_preprocessing(x, t):

    #x, mean, std_dev = normalize(x)
    x, trend_model = remove_linear_trend(x, t)

    # Turn into points
    ts = []
    for i in range(len(x)):
        ts.append(Point(x[i], t[i], "A", "A"))

    # Split time series
    ts_train = [point for point in ts if point.time_stamp <= 100]
    ts_test = [point for point in ts if point.time_stamp > 100]

    # Train model
    model = EDM()
    model.train(ts_train, max_dim=5)

    # Predict test points
    simplex, smap = model.predict(ts_test, hor=1)

    # Add back trend
    simplex['obs'] = add_linear_trend(trend_model, simplex['obs'], simplex['time_stamp'])
    simplex['pred'] = add_linear_trend(trend_model, simplex['pred'], simplex['time_stamp'])
    smap['obs'] = add_linear_trend(trend_model, smap['obs'], smap['time_stamp'])
    smap['pred'] = add_linear_trend(trend_model, smap['pred'], smap['time_stamp'])

    # # Add back mean and variance
    # simplex['obs'] = reverse_normalization(simplex['obs'], mean, std_dev)
    # simplex['pred'] = reverse_normalization(simplex['pred'], mean, std_dev)
    # smap['obs'] = reverse_normalization(smap['obs'], mean, std_dev)
    # smap['pred'] = reverse_normalization(smap['pred'], mean, std_dev)

    # Calculate performance measures
    simplex_perf = calculate_performance(simplex)
    smap_perf = calculate_performance(smap)

    return simplex_perf, smap_perf


def test_without_preprocessing(x, t):

    #x, trend_model = remove_linear_trend(x, t)

    # Turn into points
    ts = []
    for i in range(len(x)):
        ts.append(Point(x[i], t[i], "A", "A"))

    # Split time series
    ts_train = [point for point in ts if point.time_stamp <= 100]
    ts_test = [point for point in ts if point.time_stamp > 100]

    # Train model
    model = EDM()
    model.train(ts_train, max_dim=5)

    # Predict test points
    simplex, smap = model.predict(ts_test, hor=1)

    # # Add back trend
    # simplex['obs'] = add_linear_trend(trend_model, simplex['obs'], simplex['time_stamp'])
    # simplex['pred'] = add_linear_trend(trend_model, simplex['pred'], simplex['time_stamp'])
    # smap['obs'] = add_linear_trend(trend_model, smap['obs'], smap['time_stamp'])
    # smap['pred'] = add_linear_trend(trend_model, smap['pred'], smap['time_stamp'])

    # Calculate performance measures
    simplex_perf = calculate_performance(simplex)
    smap_perf = calculate_performance(smap)

    return simplex_perf, smap_perf


if __name__ == "__main__":

    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select 50 different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=50)

    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    simplex_results = pd.DataFrame(columns=range(6))
    smap_results = pd.DataFrame(columns=range(6))

    for i in range(50):

        x_, t_ = sample_lorenz(initial_vecs[i], [10, 28, 8/3], 150, 0)

        simplex_1, smap_1 = test_without_preprocessing(x_, t_)
        simplex_2, smap_2 = test_with_preprocessing(x_, t_)

        simplex_row = pd.DataFrame([[simplex_1['RMSE'], simplex_1['MAE'], simplex_1['corr'],
                       simplex_2['RMSE'], simplex_2['MAE'], simplex_2['corr']]])

        smap_row = pd.DataFrame([[smap_1['RMSE'], smap_1['MAE'], smap_1['corr'],
                    smap_2['RMSE'], smap_2['MAE'], smap_2['corr']]])

        simplex_results = pd.concat([simplex_results, simplex_row])
        smap_results = pd.concat([smap_results, smap_row])


    print("### SIMPLEX ###")
    print("Summary statistics for RMSE without preprocessing:")
    print(simplex_results[0].describe())
    print()
    print(f"Summary statistics for RMSE with preprocessing:")
    print(simplex_results[3].describe())
    print()
    simplex_percentage = (simplex_results.iloc[:,0] < simplex_results.iloc[:,3]).mean() * 100
    print("Preprocessing gave better results in " + str(simplex_percentage) + "% of the cases.")
    print("-----------------------------------------------------------------------------------")
    print()

    print("### S-MAP ###")
    print("Summary statistics for RMSE without preprocessing:")
    print(smap_results[0].describe())
    print()
    print(f"Summary statistics for RMSE with preprocessing:")
    print(smap_results[3].describe())
    print()
    smap_percentage = (smap_results.iloc[:, 0] < smap_results.iloc[:, 3]).mean() * 100
    print("Preprocessing gave better results in " + str(smap_percentage) + "% of the cases.")

