from src.classes import *
from src.simulate_lorenz import *
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
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
    x, y, z, t = simulate_lorenz(vec_0, params, n_points * 100, n_points, obs_noise)
    x, t = sample_from_ts(x, t, sampling_interval=5, n_points=n_points)
    t = [i for i in range(len(t))]
    return x, t


def sample_multiple_initial_values(vec_0, n_points, obs_noise, var):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    for i in range(1):
        x_0 = np.random.uniform(vec_0[0] - var, vec_0[0] + var)
        y_0 = np.random.uniform(vec_0[1] - var, vec_0[1] + var)
        z_0 = np.random.uniform(vec_0[2] - var, vec_0[2] + var)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, 20, 8/3], n_points, obs_noise)

        list_x.append(x)
        list_t.append(t)

    return list_x, list_t, trends, means, std_devs


def sample_multiple_rhos(vec_0, n_points, obs_noise, var):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    i = 0
    while i < 12:
        rho = np.random.uniform(20 - var, 20 + var)
        x, t = sample_lorenz(vec_0, [10, rho, 8/3], n_points, obs_noise)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t, trends, means, std_devs


def modelling(vec_0, ts_length, test, obs_noise, var, cv, frac):

    if test == "initial":
        xs, ts, trends, means, std_devs = sample_multiple_initial_values(vec_0, ts_length + 25, obs_noise, var)
    else:
        xs, ts, trends, means, std_devs = sample_multiple_rhos(vec_0, ts_length + 25, obs_noise, var)

    # Put them together into one library
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    # Split into training and test set
    ts_train = [point for point in collection if point.time_stamp <= ts_length]
    ts_test = [point for point in collection if point.time_stamp > ts_length]

    # Train model
    model = EDM(cv_method=cv, cv_fraction=frac)
    model.train(ts_train, max_dim=10)
    simplex, smap = model.predict(ts_test, hor=1)
    del(simplex)

    # Calculate RMSEs
    smap = smap.dropna()
    results = calculate_performance(smap)

    return results



def modelling_parallel(vec):

    ts_length = 30
    test = "initial"
    obs_noise = 5.0
    var = 1.0

    results = {}
    results['LB_25'] = modelling(vec, ts_length, test, obs_noise, var, cv="LB", frac=0.25)['RMSE']
    results['LB_50'] = modelling(vec, ts_length, test, obs_noise, var, cv="LB", frac=0.50)['RMSE']
    results['RB_4']  = modelling(vec, ts_length, test, obs_noise, var, cv="RB", frac=0.25)['RMSE']
    results['RB_8']  = modelling(vec, ts_length, test, obs_noise, var, cv="RB", frac=0.125)['RMSE']
    return results


if __name__ == '__main__':

    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 20, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=250)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    LB25 = []
    LB50 = []
    RB4 = []
    RB8 = []

    # Parallelize the loop
    with Pool(processes=8) as pool:
        results = pool.map(modelling_parallel, initial_vecs)

    for result in results:
        LB25.append(result['LB_25'])
        LB50.append(result['LB_50'])
        RB4.append(result['RB_4'])
        RB8.append(result['RB_8'])

    # Check normality
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(LB25, bins=20)
    axs[0, 0].set_title("Last Block CV (25%)")

    axs[0, 1].hist(LB50, bins=20)
    axs[0, 1].set_title("Last Block CV (50%)")

    axs[1, 0].hist(RB4, bins=20)
    axs[1, 0].set_title("Rolling Base CV (4 bins)")

    axs[1, 1].hist(RB8, bins=20)
    axs[1, 1].set_title("Rolling Base CV (8 bins)")

    plt.tight_layout()
    plt.show()

    # perform repeated measures ANOVA
    data = {'RMSE': LB25 + LB50 + RB4 + RB8,
            'group': ["LB_25"] * len(LB25) + ["LB_50"] * len(LB50) +
                     ["RB_4"] * len(RB4) + ["RB_8"] * len(RB8),
            'subject': list(range(1, len(LB25) + 1)) * 4
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

