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


def sample_multiple_initial_values(vec_0, n_points, remove_trend, normalization):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    for i in range(25):
        x_0 = np.random.normal(vec_0[0], 1.0)
        y_0 = np.random.normal(vec_0[1], 1.0)
        z_0 = np.random.normal(vec_0[2], 1.0)
        x, t = sample_lorenz([x_0, y_0, z_0], [10, 28, 8/3], n_points, 1.0)

        if remove_trend:
            x, trend_model = remove_linear_trend(x, t)
            trends.append(trend_model)

        if normalization:
            x, mean, std_dev = normalize(x)
            means.append(mean)
            std_devs.append(std_dev)

        list_x.append(x)
        list_t.append(t)

    return list_x, list_t, trends, means, std_devs


def sample_multiple_rhos(vec_0, n_points, remove_trend, normalization):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    for i in range(25):
        rho = np.random.normal(28, 1.0)
        x, t = sample_lorenz(vec_0, [10, rho, 8/3], n_points, 1.0)

        if remove_trend:
            x, trend_model = remove_linear_trend(x, t)
            trends.append(trend_model)

        if normalization:
            x, mean, std_dev = normalize(x)
            means.append(mean)
            std_devs.append(std_dev)

        list_x.append(x)
        list_t.append(t)

    return list_x, list_t, trends, means, std_devs


# def modelling_initial_values(vec_0, ts_length, remove_trend, normalization):
#
#     # Sample multiple time series
#     # and save their trends, means and standard deviations
#     xs, ts, trends, means, std_devs  = sample_multiple_initial_values(vec_0, ts_length, remove_trend, normalization)
#
#     # Put them together into one library
#     collection = []
#     for i, (x, t) in enumerate(zip(xs, ts)):
#         for j in range(len(x)):
#             collection.append(Point(x[j], t[j], "A", i))
#
#     ts_train = [point for point in collection if point.time_stamp <= ts_length - 25]
#     ts_test = [point for point in collection if point.time_stamp > ts_length - 25]
#
#     model = EDM()
#     model.train(ts_train, max_dim=10)
#     _, smap = model.predict(ts_test, hor=1)
#
#     smap = smap.dropna()
#
#     # Reverse preprocessing
#     if normalization:
#         for loc in range(25):
#             mean, std_dev = means[loc], std_devs[loc]
#             loc_filter = smap['location'] == loc
#             smap.loc[loc_filter, 'obs'] = reverse_normalization(smap.loc[loc_filter, 'obs'], mean, std_dev)
#             smap.loc[loc_filter, 'pred'] = reverse_normalization(smap.loc[loc_filter, 'pred'], mean, std_dev)
#
#     if remove_trend:
#         for loc in range(25):
#             trend = trends[loc]
#             loc_filter = smap['location'] == loc
#             smap.loc[loc_filter, 'obs'] = add_linear_trend(trend, smap.loc[loc_filter, 'obs'],
#                                                            smap.loc[loc_filter, 'time_stamp'])
#             smap.loc[loc_filter, 'pred'] = add_linear_trend(trend, smap.loc[loc_filter, 'pred'],
#                                                             smap.loc[loc_filter, 'time_stamp'])
#
#     results = calculate_performance(smap)
#     return results

def modelling(vec_0, ts_length, remove_trend, normalization, test):

    if test == "rho":
        xs, ts, trends, means, std_devs  = sample_multiple_rhos(vec_0, ts_length, remove_trend, normalization)
    else:
        xs, ts, trends, means, std_devs = sample_multiple_initial_values(vec_0, ts_length, remove_trend, normalization)

    # Put them together into one library
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    ts_train = [point for point in collection if point.time_stamp <= ts_length - 25]
    ts_test = [point for point in collection if point.time_stamp > ts_length - 25]

    model = EDM()
    model.train(ts_train, max_dim=10)
    _, smap = model.predict(ts_test, hor=1)

    smap = smap.dropna()

    # Reverse preprocessing
    if normalization:
        for loc in range(25):
            mean, std_dev = means[loc], std_devs[loc]
            loc_filter = smap['location'] == loc
            smap.loc[loc_filter, 'obs'] = reverse_normalization(smap.loc[loc_filter, 'obs'], mean, std_dev)
            smap.loc[loc_filter, 'pred'] = reverse_normalization(smap.loc[loc_filter, 'pred'], mean, std_dev)

    if remove_trend:
        for loc in range(25):
            trend = trends[loc]
            loc_filter = smap['location'] == loc
            smap.loc[loc_filter, 'obs'] = add_linear_trend(trend, smap.loc[loc_filter, 'obs'],
                                                           smap.loc[loc_filter, 'time_stamp'])
            smap.loc[loc_filter, 'pred'] = add_linear_trend(trend, smap.loc[loc_filter, 'pred'],
                                                            smap.loc[loc_filter, 'time_stamp'])

    results = calculate_performance(smap)
    return results


if __name__ == "__main__":

    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select 50 different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=100)

    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    no_preprocessing = []
    trend_removed = []
    normalized = []
    trend_removed_normalized = []

    i = 0
    for vec in initial_vecs:
        i += 1
        print("Round "+ str(i) + "/100")

        result = modelling(vec, 50, True, True, "initial")
        trend_removed_normalized.append(result['RMSE'])

        result = modelling(vec, 50, True, False, "initial")
        trend_removed.append(result['RMSE'])

        result = modelling(vec, 50, False, True, "initial")
        normalized.append(result['RMSE'])

        result = modelling(vec, 50, False, False, "initial")
        no_preprocessing.append(result['RMSE'])

    # Check normality
    fix, axs = plt.subplots(2, 2)
    axs[0, 0].hist(no_preprocessing, bins=20)
    axs[0, 0].set_title("No preprocessing")
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


