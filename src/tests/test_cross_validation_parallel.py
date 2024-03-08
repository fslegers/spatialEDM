from src.classes import *
from src.simulate_lorenz import *
import numpy as np

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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
    x, y, z, t = simulate_lorenz(vec_0, params, n_points, n_points, obs_noise) #TODO: maybe tmax should be different?
    t = [int(i) for i in t]
    return x, t


def sample_multiple_initial_values(vec_0, n_points):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    for i in range(25):
        x_0 = np.random.normal(vec_0[0], 1.0)
        y_0 = np.random.normal(vec_0[1], 1.0)
        z_0 = np.random.normal(vec_0[2], 1.0)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, 28, 8/3], n_points, 3.0)

        list_x.append(x)
        list_t.append(t)

    return list_x, list_t, trends, means, std_devs


def modelling(vec_0, ts_length, cv, frac):

    xs, ts, trends, means, std_devs = sample_multiple_initial_values(vec_0, ts_length)

    # Put them together into one library
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    ts_train = [point for point in collection if point.time_stamp <= 75]
    ts_test = [point for point in collection if point.time_stamp > 75]

    model = EDM(cv_method=cv, cv_fraction=frac)
    model.train(ts_train, max_dim=10)
    simplex, smap = model.predict(ts_test, hor=2)

    smap = smap.dropna()

    results = calculate_performance(smap)
    return results


import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def modelling_parallel(vec):
    results = {}
    results['LB_25'] = modelling(vec, 100, cv="LB", frac=0.25)['RMSE']
    results['LB_50'] = modelling(vec, 100, cv="LB", frac=0.5)['RMSE']
    results['RB_4'] = modelling(vec, 100, cv="RB", frac=0.25)['RMSE']
    results['RB_8'] = modelling(vec, 100, cv="RB", frac=0.125)['RMSE']
    return results


if __name__ == '__main__':
    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=100)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    last_block_25 = []
    last_block_50 = []
    rolling_base_4 = []
    rolling_base_8 = []

    # Parallelize the loop
    with Pool(processes=8) as pool:
        results = pool.map(modelling_parallel, initial_vecs)

    for result in results:
        last_block_25.append(result['LB_25'])
        last_block_50.append(result['LB_50'])
        rolling_base_4.append(result['RB_4'])
        rolling_base_8.append(result['RB_8'])

    # Check normality
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(last_block_25, bins=20)
    axs[0, 0].set_title("Last Block CV (25%)")

    axs[0, 1].hist(last_block_50, bins=20)
    axs[0, 1].set_title("Last Block CV (50%)")

    axs[1, 0].hist(rolling_base_4, bins=20)
    axs[1, 0].set_title("Rolling Base CV (4 bins)")

    axs[1, 1].hist(rolling_base_8, bins=20)
    axs[1, 1].set_title("Rolling Base CV (8 bins)")

    plt.tight_layout()
    plt.show()

    # perform repeated measures ANOVA
    data = {'RMSE': last_block_25 + last_block_50 + rolling_base_4 + rolling_base_8,
            'group': ["LB_25"] * len(last_block_25) + ["LB_50"] * len(last_block_50) +
                     ["RB_4"] * len(rolling_base_4) + ["RB_8"] * len(rolling_base_8),
            'subject': list(range(1, len(last_block_25) + 1)) * 4
            }
    df = pd.DataFrame(data)

    # Fit repeated measures ANOVA model
    model = ols('RMSE ~ C(group) + C(subject)', data=df).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # Perform pairwise post-hoc tests (e.g., Tukey HSD)
    tukey_results = pairwise_tukeyhsd(endog=df['RMSE'], groups=df['group'])
    print(tukey_results)

