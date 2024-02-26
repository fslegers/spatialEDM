import matplotlib.pyplot as plt
import pandas as pd

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


def modelling(x, t, remove_trend, normalization):

    if remove_trend:
        x, trend_model = remove_linear_trend(x, t)

    if normalization:
        x, mean, std_dev = normalize(x)

    # Turn into points
    ts = []
    for i in range(len(x)):
        ts.append(Point(x[i], t[i], "A", "A"))

    # Split time series
    ts_train = [point for point in ts if point.time_stamp <= 25]
    ts_test = [point for point in ts if point.time_stamp > 25]

    # Train model
    model = EDM()
    model.train(ts_train, max_dim=5)

    # Predict test points
    _, smap = model.predict(ts_test, hor=1)

    if normalization:
        # Add back mean and variance
        smap['obs'] = reverse_normalization(smap['obs'], mean, std_dev)
        smap['pred'] = reverse_normalization(smap['pred'], mean, std_dev)

    if remove_trend:
        # Add back trend
        smap['obs'] = add_linear_trend(trend_model, smap['obs'], smap['time_stamp'])
        smap['pred'] = add_linear_trend(trend_model, smap['pred'], smap['time_stamp'])


    # Calculate performance measures
    result = calculate_performance(smap)

    return result['RMSE']


if __name__ == "__main__":

    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select 50 different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=25) #TODO: CHANGE BACK TO 250

    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    no_preprocessing = []
    trend_removed = []
    normalized = []
    trend_removed_normalized = []

    for i in range(25): #TODO: CHANGE BACK TO 250

        x, t = sample_lorenz(initial_vecs[i], [10, 28, 8/3], 50, 0)

        no_preprocessing.append(modelling(x, t, False, False))
        trend_removed.append(modelling(x, t, True, False))
        normalized.append(modelling(x, t, False, True))
        trend_removed_normalized.append(modelling(x, t, True, True))

    # Check normality
    fix, axs = plt.subplots(2, 2)
    axs[0,0].hist(no_preprocessing, bins=20)
    axs[0,0].set_title("No preprocessing")
    axs[0, 1].hist(trend_removed_normalized, bins=20)
    axs[0, 1].set_title("Trend removed and normalized")
    axs[1, 0].hist(trend_removed, bins=20)
    axs[1, 0].set_title("Trend removed")
    axs[1, 1].hist(normalized, bins=20)
    axs[1, 1].set_title("Normalized")
    plt.tight_layout()
    plt.show()

    # Do repeated measures ANOVA
    data = {'RMSE': no_preprocessing + trend_removed + normalized + trend_removed_normalized,
            'group': ["none"] * len(no_preprocessing) + ["trend"] * len(trend_removed) +
                     ["norm"] * len(normalized) + ["both"] * len(trend_removed_normalized),
            'subject': list(range(1, len(normalized) + 1)) * 4
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


