from src.classes import *
from src.simulate_lorenz import *
import numpy as np
from tqdm import tqdm

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from multiprocessing import Pool


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


def sample_multiple_rhos(vec_0, n_points, remove_trend, normalization, n_repl, obs_noise, var):

    list_x, list_t, trends, means, std_devs = [], [], [], [], []

    i = 0
    while i < n_repl:
        rho = np.random.normal(28, var)
        x, t = sample_lorenz(vec_0, [10, rho, 8/3], n_points, obs_noise)

        if remove_trend:
            x, trend_model = remove_linear_trend(x, t)
            trends.append(trend_model)

        if normalization:
            x, mean, std_dev = normalize(x)
            means.append(mean)
            std_devs.append(std_dev)

        list_x.append(x)
        list_t.append(t)

        i += 1

    return list_x, list_t, trends, means, std_devs


def modelling(vec_0, train_length, remove_trend, normalization, n_repl, obs_noise, var):

    xs, ts, trends, means, std_devs = sample_multiple_rhos(vec_0, train_length+25, remove_trend, normalization, n_repl, obs_noise, var)

    # Put them together into one library
    collection = []
    for i, (x, t) in enumerate(zip(xs, ts)):
        for j in range(len(x)):
            collection.append(Point(x[j], t[j], "A", i))

    ts_train = [point for point in collection if point.time_stamp < train_length]
    ts_test = [point for point in collection if point.time_stamp >= train_length]

    model = EDM()
    model.train(ts_train, max_dim=10)
    simplex, smap = model.predict(ts_test, hor=1)

    smap = smap.dropna()

    # Reverse preprocessing
    if normalization:
        loc = 0
        while loc < n_repl:
            mean, std_dev = means[loc], std_devs[loc]
            loc_filter = smap['location'] == loc
            smap.loc[loc_filter, 'obs'] = reverse_normalization(smap.loc[loc_filter, 'obs'], mean, std_dev)
            smap.loc[loc_filter, 'pred'] = reverse_normalization(smap.loc[loc_filter, 'pred'], mean, std_dev)
            loc += 1

    if remove_trend:
        loc = 0
        while loc < n_repl:
            trend = trends[loc]
            loc_filter = smap['location'] == loc
            try:
                smap.loc[loc_filter, 'obs'] = add_linear_trend(trend, smap.loc[loc_filter, 'obs'],
                                                               smap.loc[loc_filter, 'time_stamp'])
                smap.loc[loc_filter, 'pred'] = add_linear_trend(trend, smap.loc[loc_filter, 'pred'],
                                                                smap.loc[loc_filter, 'time_stamp'])
            except:
                print("...")
            loc += 1

    results = calculate_performance(smap)
    return results


def modelling_parallel(vec):
    results = {}

    train_length = 50
    n_replicates = 12
    obs_noise = 1.0
    var = 1.0

    results['FF'] = modelling(vec, train_length, False, False, n_replicates, obs_noise, var)['RMSE']
    results['TF'] = modelling(vec, train_length, True, False, n_replicates, obs_noise, var)['RMSE']
    results['FT'] = modelling(vec, train_length, False, True, n_replicates, obs_noise, var)['RMSE']
    results['TT'] = modelling(vec, train_length, True, True, n_replicates, obs_noise, var)['RMSE']
    return results


if __name__ == '__main__':
    # simulate one big giant Lorenz attractor without transient
    x, y, z, t = simulate_lorenz([1, 1, 1], [10, 28, 8 / 3], 2000, 35, 0)
    x, y, z = x[1000:], y[1000:], z[1000:]

    # select different initial points from this trajectory
    initial_vecs = []
    indices = np.random.randint(len(x), size=250)
    for i in indices:
        initial_vecs.append([x[i], y[i], z[i]])

    # Prepare to save results
    no_preprocessing = []
    trend_removed = []
    normalized = []
    trend_removed_normalized = []

    print("--- Starting main loop ---")

    # Parallelize the loop
    with Pool(processes=10) as pool:
        results = list(tqdm(pool.map(modelling_parallel, initial_vecs), total=len(initial_vecs)))

    for result in results:
        no_preprocessing.append(result['FF'])
        trend_removed.append(result['TF'])
        normalized.append(result['FT'])
        trend_removed_normalized.append(result['TT'])

    # Check normality
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].hist(no_preprocessing, bins=30)
    axs[0, 0].set_title("No preprocessing")

    axs[0, 1].hist(trend_removed, bins=30)
    axs[0, 1].set_title("Trend removed")

    axs[1, 0].hist(normalized, bins=30)
    axs[1, 0].set_title("Normalization")

    axs[1, 1].hist(trend_removed_normalized, bins=30)
    axs[1, 1].set_title("Trend removed normalized")

    plt.tight_layout()
    plt.show()

    # perform repeated measures ANOVA
    data = {'RMSE': no_preprocessing + trend_removed + normalized + trend_removed_normalized,
            'group': ["FF"] * len(no_preprocessing) + ["TF"] * len(trend_removed) +
                     ["FT"] * len(normalized) + ["TT"] * len(trend_removed_normalized),
            'subject': list(range(1, len(no_preprocessing) + 1)) * 4
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

