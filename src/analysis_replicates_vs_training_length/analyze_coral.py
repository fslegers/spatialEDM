import time
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
from src.classes import *
from src.simulate_lorenz import *
import pandas as pd
from matplotlib import cm
import os
from brokenaxes import brokenaxes


def interpolate_missing_values(values, timestamps, target_interval):
    # Convert timestamps to numpy array
    timestamps = np.array(timestamps)

    # Calculate the new timestamps with regular intervals
    new_timestamps = np.arange(timestamps[0], timestamps[-1] + target_interval, target_interval)

    # Use numpy's interpolation function to fill in missing values
    interpolated_values = np.interp(new_timestamps, timestamps, values)

    return interpolated_values, new_timestamps


def local_level(df):

    simplex = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])
    smap = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])

    for loc in df['location']:
        time.sleep(1)
        try:

            df_sub = df[(df['location'] == loc)]
            hab_ = df_sub['habitat'].values[0]
            site_ = df_sub['site'].values[0]
            trans_ = df_sub['transect'].values[0]
            quad_ = df_sub['quadrat'].values[0]

            ts = df_sub['ts'].values[0]
            ts_train = [point for point in ts if point.time_stamp < 2020]
            ts_test = [point for point in ts if point.time_stamp >= 2020]

            model = EDM()
            model.train(ts_train, max_dim=5)

            simplex_loc, smap_loc = model.predict(ts_test, 1)

            for i in range(len(simplex_loc['pred'])):
                if not math.isnan(simplex_loc['obs'][i]):

                    simplex_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': simplex_loc['obs'][i],
                                   'pred': simplex_loc['pred'][i]}
                    simplex = pd.concat([simplex, pd.DataFrame([simplex_row])], ignore_index=True)

                    smap_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': smap_loc['obs'][i],
                                'pred': smap_loc['pred'][i]}
                    smap = pd.concat([smap, pd.DataFrame([smap_row])], ignore_index=True)

        except:
            print(loc)

    return simplex, smap


def transect_level(df):

    simplex = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])
    smap = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])

    for trans in list(set(df['transect'])):

        df_sub = df[(df['transect'] == trans)][['ts']]

        df_sub = df_sub.explode('ts', ignore_index=True)
        ts = df_sub['ts'].values

        ts_train = [point for point in ts if point.time_stamp < 2020]
        ts_test = [point for point in ts if point.time_stamp >= 2020]

        model = EDM()
        model.train(ts_train, max_dim=5)

        simplex_loc, smap_loc = model.predict(ts_test, 1)

        for i in range(len(simplex_loc['pred'])):
            if not math.isnan(simplex_loc['obs'][i]):

                loc_ = simplex_loc['location'][i]
                hab_ = df[df['location'] == loc_]['habitat'].values[0]
                site_ = df[df['location'] == loc_]['site'].values[0]
                trans_ = df[df['location'] == loc_]['transect'].values[0]
                quad_ = df[df['location'] == loc_]['quadrat'].values[0]

                simplex_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': simplex_loc['obs'][i],
                               'pred': simplex_loc['pred'][i]}
                simplex = pd.concat([simplex, pd.DataFrame([simplex_row])], ignore_index=True)

                smap_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': smap_loc['obs'][i],
                               'pred': smap_loc['pred'][i]}
                smap = pd.concat([smap, pd.DataFrame([smap_row])], ignore_index=True)

    return simplex, smap


def site_level(df):

    simplex = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])
    smap = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])

    for site in list(set(df['site'])):

        df_sub = df[(df['site'] == site)]

        df_sub = df_sub.explode('ts', ignore_index=True)
        ts = df_sub['ts'].values

        ts_train = [point for point in ts if point.time_stamp < 2020]
        ts_test = [point for point in ts if point.time_stamp >= 2020]

        model = EDM()
        model.train(ts_train, max_dim=5)

        simplex_loc, smap_loc = model.predict(ts_test, 1)

        for i in range(len(simplex_loc['pred'])):
            if not math.isnan(simplex_loc['obs'][i]):

                loc_ = simplex_loc['location'][i]
                hab_ = df[df['location'] == loc_]['habitat'].values[0]
                site_ = df[df['location'] == loc_]['site'].values[0]
                trans_ = df[df['location'] == loc_]['transect'].values[0]
                quad_ = df[df['location'] == loc_]['quadrat'].values[0]

                simplex_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': simplex_loc['obs'][i],
                               'pred': simplex_loc['pred'][i]}
                simplex = pd.concat([simplex, pd.DataFrame([simplex_row])], ignore_index=True)

                smap_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': smap_loc['obs'][i],
                            'pred': smap_loc['pred'][i]}
                smap = pd.concat([smap, pd.DataFrame([smap_row])], ignore_index=True)

    return simplex, smap


def global_level(df):

    simplex = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])
    smap = pd.DataFrame(columns=['habitat', 'site', 'transect', 'quadrat', 'obs', 'pred'])

    df = df.explode('ts', ignore_index=True)
    ts = df['ts'].values

    ts_train = [point for point in ts if point.time_stamp < 2020]
    ts_test = [point for point in ts if point.time_stamp >= 2020]

    model = EDM()
    model.train(ts_train, max_dim=5)

    simplex_loc, smap_loc = model.predict(ts_test, 1)

    for i in range(len(simplex_loc['pred'])):
        if not math.isnan(simplex_loc['obs'][i]):

            loc_ = simplex_loc['location'][i]
            hab_ = df[df['location'] == loc_]['habitat'].values[0]
            site_ = df[df['location'] == loc_]['site'].values[0]
            trans_ = df[df['location'] == loc_]['transect'].values[0]
            quad_ = df[df['location'] == loc_]['quadrat'].values[0]

            simplex_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': simplex_loc['obs'][i],
                           'pred': simplex_loc['pred'][i]}
            simplex = pd.concat([simplex, pd.DataFrame([simplex_row])], ignore_index=True)

            smap_row = {'habitat': hab_, 'site': site_, 'transect': trans_, 'quadrat': quad_, 'obs': smap_loc['obs'][i],
                        'pred': smap_loc['pred'][i]}
            smap = pd.concat([smap, pd.DataFrame([smap_row])], ignore_index=True)

    return simplex, smap


def make_backreef_data_set(species ="Echinometra mathaei", habitat ="Backreef"):
    df = pd.read_csv('C:/Users/5605407/Documents/PhD/Data sets/LTER Coral Reef/dt1.csv', delimiter=',', header=0)
    df = df[(df['Taxonomy'] == species)]
    df = df[(df['Habitat'] == habitat)]

    # Leave out data from 2005, as measurements were taken in may instead of january.
    df = df[df['Year'] > 2005]

    # Save time series for each location
    df_ts = pd.DataFrame(columns=['location', 'habitat', 'site', 'transect', 'quadrat', 'ts'])

    for loc in set(df['Location']):

        time.sleep(1)

        df_subset = df[df['Location'] == loc]

        time_stamps = (df_subset['Year'].values).tolist()
        values = df_subset['Count'].values

        if len(time_stamps) >= 11: # Only take time series that have enough observations (11)

            values_, time_stamps_ = interpolate_missing_values(values, time_stamps, 1)
            ts = transform_array_to_ts(values_, time_stamps_, loc=loc, spec=species)

            # Problematic time series (17)
            problematic = ["LTER 3 Backreef Invertebrate Herbivores Transect 2 Quad 3",
                           "LTER 6 Backreef Invertebrate Herbivores Transect 2 Quad 3",
                           "LTER 2 Backreef Invertebrate Herbivores Transect 2 Quad 4",
                           "LTER 5 Backreef Invertebrate Herbivores Transect 5 Quad 4",
                           "LTER 4 Backreef Invertebrate Herbivores Transect 5 Quad 2",
                           "LTER 2 Backreef Invertebrate Herbivores Transect 1 Quad 3",
                           "LTER 1 Backreef Invertebrate Herbivores Transect 1 Quad 2",
                           "LTER 5 Backreef Invertebrate Herbivores Transect 3 Quad 3",
                           "LTER 1 Backreef Invertebrate Herbivores Transect 1 Quad 4",
                           "LTER 5 Backreef Invertebrate Herbivores Transect 3 Quad 2",
                           "LTER 1 Backreef Invertebrate Herbivores Transect 5 Quad 4",
                           "LTER 3 Backreef Invertebrate Herbivores Transect 1 Quad 2",
                           "LTER 5 Backreef Invertebrate Herbivores Transect 5 Quad 1",
                           "LTER 5 Backreef Invertebrate Herbivores Transect 2 Quad 4",
                           "LTER 1 Backreef Invertebrate Herbivores Transect 2 Quad 1",
                           "LTER 6 Backreef Invertebrate Herbivores Transect 1 Quad 4",
                           "LTER 6 Backreef Invertebrate Herbivores Transect 1 Quad 3"]

            # TODO: Normalize/standardize?
            # For now, I'm not yet convinced that we should do this

            if loc not in problematic:

                # TODO: Remove trend/seasonality

                site = str(df_subset['Site'].values[0])
                transect = str(df_subset['Transect'].values[0])
                quadrat = str(df_subset['Quadrat'].values[0])

                new_row = {'location': loc, 'habitat': habitat, 'site': site, 'transect': transect, 'quadrat': quadrat, 'ts': ts}
                df_ts = pd.concat([df_ts, pd.DataFrame([new_row])], ignore_index=True)

    df = df_ts
    del(df_ts)

    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    simplex_local, smap_local = local_level(df)
    simplex_local.to_csv("./simplex_local_backreef.csv")
    smap_local.to_csv("./smap_local_backreef.csv")

    simplex_transect, smap_transect = transect_level(df)
    simplex_transect.to_csv("./simplex_transect_backreef.csv")
    smap_transect.to_csv("./smap_transect_backreef.csv")

    simplex_site, smap_site = site_level(df)
    simplex_site.to_csv("./simplex_site_backreef.csv")
    smap_site.to_csv("./smap_site_backreef.csv")

    simplex_global, smap_global = global_level(df)
    simplex_global.to_csv("./simplex_global_backreef.csv")
    smap_global.to_csv("./smap_global_backreef.csv")


def make_fringing_data_set(species ="Echinometra mathaei", habitat ="Fringing"):

    # Problematic time series
    problematic = ["LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 3",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 3 Fringing Reef Invertebrate Herbivores Transect 4 Quad 3",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 1",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 3 Quad 1",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 5 Quad 3",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 2 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 4 Quad 4",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 4 Quad 3",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 2 Quad 4",
                   "LTER 4 Fringing Reef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 1 Fringing Reef Invertebrate Herbivores Transect 4 Quad 1",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 5 Quad 1",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 2 Quad 2",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 2",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 1 Quad 1",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 2 Quad 2",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 3 Quad 4",
                   "LTER 1 Fringing Reef Invertebrate Herbivores Transect 5 Quad 4",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 5 Quad 1",
                   "LTER 3 Fringing Reef Invertebrate Herbivores Transect 2 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 5 Quad 4",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 2",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 4 Quad 3",
                   "LTER 3 Fringing Reef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 1",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 2 Quad 1",
                   "LTER 1 Fringing Reef Invertebrate Herbivores Transect 2 Quad 2",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 2 Quad 1"]

    # Load data
    df = pd.read_csv('C:/Users/5605407/Documents/PhD/Data sets/LTER Coral Reef/dt1.csv', delimiter=',', header=0)

    df = df[(df['Taxonomy'] == species)]
    df = df[(df['Habitat'] == "Fringing")]

    # Leave out data from 2005, as measurements were taken in may instead of january.
    df = df[df['Year'] > 2005]

    # Save time series for each location
    df_ts = pd.DataFrame(columns=['location', 'habitat', 'site', 'transect', 'quadrat', 'ts'])

    for loc in set(df['Location']):

        time.sleep(1)

        if loc not in problematic:
            df_subset = df[df['Location'] == loc]
            time_stamps = (df_subset['Year'].values).tolist()
            values = df_subset['Count'].values

            if len(time_stamps) >= 11:

                values_, time_stamps_ = interpolate_missing_values(values, time_stamps, 1)
                ts = transform_array_to_ts(values_, time_stamps_, loc=loc, spec=species)\

                if loc not in problematic:

                    site = str(df_subset['Site'].values[0])
                    transect = str(df_subset['Transect'].values[0])
                    quadrat = str(df_subset['Quadrat'].values[0])

                    new_row = {'location': loc, 'habitat': habitat, 'site': site, 'transect': transect, 'quadrat': quadrat, 'ts': ts}
                    df_ts = pd.concat([df_ts, pd.DataFrame([new_row])], ignore_index=True)

    df = df_ts
    del(df_ts)

    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    simplex_local, smap_local = local_level(df)
    simplex_local.to_csv("./simplex_local_fringing.csv")
    smap_local.to_csv("./smap_local_fringing.csv")

    simplex_transect, smap_transect = transect_level(df)
    simplex_transect.to_csv("./simplex_transect_fringing.csv")
    smap_transect.to_csv("./smap_transect_fringing.csv")

    simplex_site, smap_site = site_level(df)
    simplex_site.to_csv("./simplex_site_fringing.csv")
    smap_site.to_csv("./smap_site_fringing.csv")

    simplex_global, smap_global = global_level(df)
    simplex_global.to_csv("./simplex_global_fringing.csv")
    smap_global.to_csv("./smap_global_fringing.csv")


def make_combined_data_set(species ="Echinometra mathaei"):

    # Problematic time series
    problematic = ["LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 3",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 3 Fringing Reef Invertebrate Herbivores Transect 4 Quad 3",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 1",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 3 Quad 1",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 5 Quad 3",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 2 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 4 Quad 4",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 4 Quad 3",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 2 Quad 4",
                   "LTER 4 Fringing Reef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 1 Fringing Reef Invertebrate Herbivores Transect 4 Quad 1",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 5 Quad 1",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 2 Quad 2",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 1 Quad 2",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 1 Quad 1",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 2 Quad 2",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 3 Quad 4",
                   "LTER 1 Fringing Reef Invertebrate Herbivores Transect 5 Quad 4",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 5 Quad 1",
                   "LTER 3 Fringing Reef Invertebrate Herbivores Transect 2 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 5 Quad 4",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 2",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 4 Quad 3",
                   "LTER 3 Fringing Reef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 2 Fringing Reef Invertebrate Herbivores Transect 1 Quad 1",
                   "LTER 6 Fringing Reef Invertebrate Herbivores Transect 2 Quad 1",
                   "LTER 1 Fringing Reef Invertebrate Herbivores Transect 2 Quad 2",
                   "LTER 5 Fringing Reef Invertebrate Herbivores Transect 2 Quad 1",
                   "LTER 3 Backreef Invertebrate Herbivores Transect 2 Quad 3",
                   "LTER 6 Backreef Invertebrate Herbivores Transect 2 Quad 3",
                   "LTER 2 Backreef Invertebrate Herbivores Transect 2 Quad 4",
                   "LTER 5 Backreef Invertebrate Herbivores Transect 5 Quad 4",
                   "LTER 4 Backreef Invertebrate Herbivores Transect 5 Quad 2",
                   "LTER 2 Backreef Invertebrate Herbivores Transect 1 Quad 3",
                   "LTER 1 Backreef Invertebrate Herbivores Transect 1 Quad 2",
                   "LTER 5 Backreef Invertebrate Herbivores Transect 3 Quad 3",
                   "LTER 1 Backreef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 5 Backreef Invertebrate Herbivores Transect 3 Quad 2",
                   "LTER 1 Backreef Invertebrate Herbivores Transect 5 Quad 4",
                   "LTER 3 Backreef Invertebrate Herbivores Transect 1 Quad 2",
                   "LTER 5 Backreef Invertebrate Herbivores Transect 5 Quad 1",
                   "LTER 5 Backreef Invertebrate Herbivores Transect 2 Quad 4",
                   "LTER 1 Backreef Invertebrate Herbivores Transect 2 Quad 1",
                   "LTER 6 Backreef Invertebrate Herbivores Transect 1 Quad 4",
                   "LTER 6 Backreef Invertebrate Herbivores Transect 1 Quad 3"]

    # Load data
    df = pd.read_csv('C:/Users/5605407/Documents/PhD/Data sets/LTER Coral Reef/dt1.csv', delimiter=',', header=0)

    df = df[(df['Taxonomy'] == species)]
    df = df[(df['Habitat'] == "Backreef") |  (df['Habitat'] == "Fringing")]

    # Leave out data from 2005, as measurements were taken in may instead of january.
    df = df[df['Year'] > 2005]

    # Save time series for each location
    df_ts = pd.DataFrame(columns=['location', 'habitat', 'site', 'transect', 'quadrat', 'ts'])

    for loc in set(df['Location']):

        time.sleep(1)

        if loc not in problematic:
            df_subset = df[df['Location'] == loc]
            time_stamps = (df_subset['Year'].values).tolist()
            values = df_subset['Count'].values

            if len(time_stamps) >= 11:

                values_, time_stamps_ = interpolate_missing_values(values, time_stamps, 1)
                ts = transform_array_to_ts(values_, time_stamps_, loc=loc, spec=species)\

                if loc not in problematic:

                    habitat = df_subset['Habitat'].values[0]
                    site = str(df_subset['Site'].values[0])
                    transect = str(df_subset['Transect'].values[0])
                    quadrat = str(df_subset['Quadrat'].values[0])

                    new_row = {'location': loc, 'habitat': habitat, 'site': site, 'transect': transect, 'quadrat': quadrat, 'ts': ts}
                    df_ts = pd.concat([df_ts, pd.DataFrame([new_row])], ignore_index=True)

    df = df_ts
    del(df_ts)

    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    simplex_local, smap_local = local_level(df)
    simplex_local.to_csv("./simplex_local_combined.csv")
    smap_local.to_csv("./smap_local_combined.csv")

    simplex_transect, smap_transect = transect_level(df)
    simplex_transect.to_csv("./simplex_transect_combined.csv")
    smap_transect.to_csv("./smap_transect_combined.csv")

    simplex_site, smap_site = site_level(df)
    simplex_site.to_csv("./simplex_site_combined.csv")
    smap_site.to_csv("./smap_site_combined.csv")

    simplex_global, smap_global = global_level(df)
    simplex_global.to_csv("./simplex_global_combined.csv")
    smap_global.to_csv("./smap_global_combined.csv")


def make_both_plots(simplex_local, simplex_transect, simplex_site, simplex_global, smap_local, smap_transect, smap_site, smap_global):

    # Create a list of colors for the different sites
    cmap = plt.get_cmap('tab20b')
    norm = plt.Normalize(0, 5)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [scalar_map.to_rgba(value) for value in range(6)]

    ### Make plot for each level ###
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))

    color_list = []
    for i in range(len(simplex_local)):
        which_site = simplex_local['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    # Local
    axs[0, 0].scatter(simplex_local['obs'], simplex_local['pred'], zorder=1, color=color_list, alpha=.7)
    axs[1, 0].scatter(smap_local['obs'], smap_local['pred'], zorder=1, color=color_list, alpha=.7)

    r, p = pearsonr(simplex_local['obs'], simplex_local['pred'])
    axs[0, 0].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[0, 0].transAxes)

    r, p = pearsonr(smap_local['obs'], smap_local['pred'])
    axs[1, 0].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[1, 0].transAxes)

    # Transect-level

    color_list = []
    for i in range(len(simplex_transect)):
        which_site = simplex_transect['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    axs[0, 1].scatter(simplex_transect['obs'], simplex_transect['pred'], zorder=1, color=color_list, alpha=.7)
    axs[1, 1].scatter(smap_transect['obs'], smap_transect['pred'], zorder=1, color=color_list, alpha=.7)

    r, p = pearsonr(simplex_transect['obs'], simplex_transect['pred'])
    axs[0, 1].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[0, 1].transAxes)

    r, p = pearsonr(smap_transect['obs'], smap_transect['pred'])
    axs[1, 1].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[1, 1].transAxes)

    # Site-level
    color_list = []
    for i in range(len(simplex_site)):
        which_site = simplex_site['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    axs[0, 2].scatter(simplex_site['obs'], simplex_site['pred'], zorder=1, color=color_list, alpha=.7)
    axs[1, 2].scatter(smap_site['obs'], smap_site['pred'], zorder=1, color=color_list, alpha=.7)

    r, p = pearsonr(simplex_site['obs'], simplex_site['pred'])
    axs[0, 2].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[0, 2].transAxes)

    r, p = pearsonr(smap_site['obs'], smap_site['pred'])
    axs[1, 2].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[1, 2].transAxes)

    # Global-level
    color_list = []
    for i in range(len(simplex_global)):
        which_site = simplex_global['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    axs[0, 3].scatter(simplex_global['obs'], simplex_global['pred'], zorder=1, color=color_list, alpha=.7)
    axs[1, 3].scatter(smap_global['obs'], smap_global['pred'], zorder=1, color=color_list, alpha=.7)

    r, p = pearsonr(simplex_global['obs'], simplex_global['pred'])
    axs[0, 3].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[0, 3].transAxes)

    r, p = pearsonr(smap_global['obs'], smap_global['pred'])
    axs[1, 3].text(.3, 0.9, 'ρ={:.3f}'.format(r), fontsize=14, transform=axs[1, 3].transAxes)

    min_1 = min(simplex_local['obs'] + simplex_local['pred']) - 0.1
    max_1 = max(simplex_local['obs'] + simplex_local['pred']) + 0.1

    min_2 = min(smap_local['obs'] + smap_local['pred']) - 0.1
    max_2 = max(smap_local['obs'] + smap_local['pred']) + 0.1

    for i in range(2):
        for j in range(4):

            if i == 0:
                axs[i, j].plot([min_1, max_1], [min_1, max_1], zorder=0, color='black')

            else:
                axs[i, j].plot([min_2, max_2], [min_2, max_2], zorder=0, color='black')

            axs[i, j].set_xlabel("observed", fontsize=12)
            axs[i, j].set_ylabel("predicted", fontsize=12)
            axs[i, j].yaxis.set_major_locator(MaxNLocator(nbins=3))
            axs[i, j].xaxis.set_major_locator(MaxNLocator(nbins=3))
            axs[i, j].set_aspect('equal')
            axs[i, j].set_aspect('equal')

    axs[0, 0].set_title("Local", fontsize=17)
    axs[0, 1].set_title("Transect-level", fontsize=17)
    axs[0, 2].set_title("Site-level", fontsize=17)
    axs[0, 3].set_title("Global", fontsize=17)

    # Plot
    plt.tight_layout()
    plt.show()


def make_backreef_boxplots(all_results):

    for df in all_results:
        df['AE'] = np.abs(df['obs'] - df['pred'])

    levels = ['Local\nlevel', 'Transect\nlevel', 'Site\nlevel', 'Global\nlevel']

    color_palette = plt.cm.tab10
    palette = [color_palette(value) for value in np.linspace(0, 1, 6)]

    # Create boxplot in each subplot
    all_data = []
    for i in range(len(all_results)):
        all_data.append(all_results[i]["AE"])

    # Make one boxplot
    # plt.boxplot(all_data, showmeans=True, meanline=True, labels=levels, showfliers=False)
    boxprops = dict(linestyle='-', linewidth=2.5, facecolor='w', alpha=0.8, edgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    meanlineprops = dict(linestyle=(0, (1,1)), linewidth=2.5, color='black')
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)

    bax = brokenaxes(ylims=((-0.01, 0.37), (0.49, 0.51), (1.07, 1.1)))
    bax.boxplot(all_data, showmeans=True, meanline=True, labels=levels, showfliers=False, boxprops=boxprops,
                medianprops=medianprops, meanprops=meanlineprops, whiskerprops=whiskerprops, capprops=capprops,
                widths=.75, patch_artist=True)

    # Add colored data points
    sites_placement = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]
    for i in range(len(all_results)):
        x = []
        colors = []
        for j in range(len(all_results[i]['AE'])):
            which_site = all_results[i]['site'].values[j]
            which_site = which_site.replace("LTER ", "")
            which_site = int(which_site) - 1
            colors.append(palette[which_site])
            x.append(i + 1 + sites_placement[which_site])
        bax.scatter(x, all_results[i]['AE'], alpha=0.8, color=colors)

    bax.set_xticks(np.arange(1, len(levels) + 1), labels=levels)
    plt.savefig('./Boxplots Backreef Echinometra mathaei.png', dpi=1200)
    plt.show()

    return 0


def make_fringing_boxplots(all_results):

    for df in all_results:
        df['AE'] = np.abs(df['obs'] - df['pred'])

    levels = ['Local\nlevel', 'Transect\nlevel', 'Site\nlevel', 'Global\nlevel']

    color_palette = plt.cm.tab10
    palette = [color_palette(value) for value in np.linspace(0, 1, 6)]

    # Create boxplot in each subplot
    all_data = []
    for i in range(len(all_results)):
        all_data.append(all_results[i]["AE"])

    # Make one boxplot
    # plt.boxplot(all_data, showmeans=True, meanline=True, labels=levels, showfliers=False)
    boxprops = dict(linestyle='-', linewidth=2.5, facecolor='w', alpha=0.8, edgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    meanlineprops = dict(linestyle=(0, (1,1)), linewidth=2.5, color='black')
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)

    bax = brokenaxes(ylims=((-0.01, 0.37), (0.49, 0.51), (1.07, 1.1)))
    bax.boxplot(all_data, showmeans=True, meanline=True, labels=levels, showfliers=False, boxprops=boxprops,
                medianprops=medianprops, meanprops=meanlineprops, whiskerprops=whiskerprops, capprops=capprops,
                widths=.75, patch_artist=True)

    # Add colored data points
    sites_placement = [-0.25, -0.15, -0.05, 0.05, 0.15, 0.25]
    for i in range(len(all_results)):
        x = []
        colors = []
        for j in range(len(all_results[i]['AE'])):
            which_site = all_results[i]['site'].values[j]
            which_site = which_site.replace("LTER ", "")
            which_site = int(which_site) - 1
            colors.append(palette[which_site])
            x.append(i + 1 + sites_placement[which_site])
        bax.scatter(x, all_results[i]['AE'], alpha=0.8, color=colors)

    bax.set_xticks(np.arange(1, len(levels) + 1), labels=levels)
    plt.savefig('./Boxplots Fringing Echinometra mathaei.png', dpi=1200)
    plt.show()

    return 0


def make_combined_boxplots(backreef, fringing, combined):

    # Add AE
    for index in range(len(backreef)):
        backreef[index]['AE'] = np.abs(backreef[index]['obs'] - backreef[index]['pred'])
    for index in range(len(fringing)):
        fringing[index]['AE'] = np.abs(fringing[index]['obs'] - fringing[index]['pred'])
    for index in range(len(combined)):
        combined[index]['AE'] = np.abs(combined[index]['obs'] - combined[index]['pred'])

    # Concatenate the backreef and fringing dataframes
    not_combined = []
    for i in range(3):
        not_combined.append(pd.concat([backreef[index], fringing[index]], axis=0, ignore_index=True))

    all_data = []
    for j in range(3):
        not_combined[index] = not_combined[index][~np.isnan(not_combined[index]['AE'])]
        combined[index] = combined[index][~np.isnan(combined[index]['AE'])]
        all_data.append(not_combined[index]['AE'].values)
        all_data.append(combined[index]['AE'].values)

    levels = ['Transect', 'Transect \n combined', 'Site', 'Site \n combined', 'Global', 'Global \n combined']

    # Define colors
    color_palette = plt.cm.tab10
    palette = [color_palette(value) for value in np.linspace(0, 1, 5)]
    placement = [0.9, 1.1]

    # Other style choices
    boxprops = dict(linestyle='-', linewidth=2.5, facecolor='w', alpha=0.8, edgecolor='black')
    medianprops = dict(linestyle='-', linewidth=2.5, color='black')
    meanlineprops = dict(linestyle=(0, (1, 1)), linewidth=2.5, color='black')
    whiskerprops = dict(linewidth=2)
    capprops = dict(linewidth=2)

    fig, bax = plt.subplots()
    bax.boxplot(all_data, showmeans=True, meanline=True, labels=levels, showfliers=False, boxprops=boxprops,
                medianprops=medianprops, meanprops=meanlineprops, whiskerprops=whiskerprops, capprops=capprops,
                widths=.5, patch_artist=True)

    # Add colored data points
    for index in range(6):
        x = []
        colors = []

        # Backreef or fringing in library
        if index in [0, 2, 4]:
            data_row = not_combined[int(index/2)]
        else:
            data_row = combined[int((index - 1)/2)]

        for index_2 in range(len(data_row)):
            if data_row['habitat'][index_2] == 'Backreef':
                x.append(index + placement[0])
                colors.append(palette[0])
            else: # habitat == Fringing
                x.append(index + placement[1])
                colors.append(palette[1])

        if index in [0,1]:
            bax.scatter(x, all_data[index], alpha=0.6, color=colors, label=data_row['habitat'][0])
        else:
            bax.scatter(x, all_data[index], alpha=0.6, color=colors)

    legend = bax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    legend.set_title('Habitat')

    bax.set_xticks(np.arange(1, len(levels) + 1), labels=levels)
    plt.tight_layout()
    plt.savefig('./Boxplots Combined Echinometra mathaei.png', dpi=1200)
    plt.show()

    return 0


def make_smap_plots(smap_local, smap_transect, smap_site, smap_global):

    #min_ = min(smap_local['obs'] + smap_local['pred'] + smap_transect['obs'] + smap_transect['pred'] + smap_site['obs'] + smap_site['pred'] + smap_global['obs'] + smap_global['pred']) - 0.1
    #max_ = max(smap_local['obs'] + smap_local['pred'] + smap_transect['obs'] + smap_transect['pred'] + smap_site['obs'] + smap_site['pred'] + smap_global['obs'] + smap_global['pred']) + 0.1

    min_ = -0.15
    max_ = 0.55

    # Create a list of colors for the different sites
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(0, 5)
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = [scalar_map.to_rgba(value) for value in range(6)]

    # Local
    color_list = []
    for i in range(len(smap_local)):
        which_site = smap_local['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    r, p = pearsonr(smap_local['obs'], smap_local['pred'])
    plt.plot([min_, max_], [min_, max_], color='black', alpha=.6, zorder=0)
    plt.scatter(smap_local['obs'], smap_local['pred'], s=70, color=color_list, alpha=.9, zorder=1)

    plt.text(0.12, 0.5, 'ρ={:.3f}'.format(r), fontsize=18)
    plt.xlabel("Observed", fontsize=18)
    plt.ylabel("Predicted", fontsize=18)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.axis('equal')
    plt.title("Local", fontsize=22)
    plt.tight_layout()
    plt.savefig("./smap_local_coral")
    plt.show()

    # Transect
    color_list = []
    for i in range(len(smap_transect)):
        which_site = smap_transect['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    r, p = pearsonr(smap_transect['obs'], smap_transect['pred'])
    plt.plot([min_, max_], [min_, max_], color='black', alpha=.6, zorder=0)
    plt.scatter(smap_transect['obs'], smap_transect['pred'], s=70, alpha=0.9, zorder=1, color=color_list)

    plt.text(0.12, 0.5, 'ρ={:.3f}'.format(r), fontsize=18)
    plt.xlabel("Observed", fontsize=18)
    plt.ylabel("Predicted", fontsize=18)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.axis('equal')
    plt.title("Transect", fontsize=22)
    plt.tight_layout()
    plt.savefig("./smap_transect_coral")
    plt.show()

    # Site
    color_list = []
    for i in range(len(smap_site)):
        which_site = smap_site['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    r, p = pearsonr(smap_site['obs'], smap_site['pred'])
    plt.plot([min_, max_], [min_, max_], zorder=0, color='black', alpha=0.6)
    plt.scatter(smap_site['obs'], smap_site['pred'], alpha=0.9, zorder=1, s=70, color=color_list)

    plt.text(0.12, 0.5, 'ρ={:.3f}'.format(r), fontsize=18)
    plt.xlabel("Observed", fontsize=18)
    plt.ylabel("Predicted", fontsize=18)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.axis('equal')
    plt.title("Site", fontsize=22)
    plt.tight_layout()
    plt.savefig("./smap_site_coral")
    plt.show()

    # Global
    color_list = []
    for i in range(len(smap_global)):
        which_site = smap_global['site'].values[i]
        which_site = which_site.replace("LTER ", "")
        which_site = int(which_site) - 1
        color_list.append(colors[which_site])

    r, p = pearsonr(smap_global['obs'], smap_global['pred'])
    plt.plot([min_, max_], [min_, max_], color='black', alpha=.6, zorder=0)
    plt.scatter(smap_global['obs'], smap_global['pred'], s=70, color=color_list, alpha=.9, zorder=1)

    plt.text(0.12, 0.5, 'ρ={:.3f}'.format(r), fontsize=18)
    plt.xlabel("Observed", fontsize=18)
    plt.ylabel("Predicted", fontsize=18)
    plt.xlim(min_, max_)
    plt.ylim(min_, max_)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    #plt.axis('equal')
    plt.title("Global", fontsize=22)
    plt.tight_layout()
    plt.savefig("./smap_global_coral")
    plt.show()


if __name__ == "__main__":

    # only run once
    # make_backreef_data_set()
    # make_fringing_data_set()
    # make_combined_data_set()

    # Change directory
    os.chdir('../..')
    os.chdir('results')
    os.chdir('output')

    # Load Backreef data
    simplex_local_backreef = pd.read_csv("./simplex_local_backreef.csv")
    smap_local_backreef = pd.read_csv("./smap_local_backreef.csv")

    simplex_transect_backreef = pd.read_csv("./simplex_transect_backreef.csv")
    smap_transect_backreef = pd.read_csv("./smap_transect_backreef.csv")

    simplex_site_backreef = pd.read_csv("./simplex_site_backreef.csv")
    smap_site_backreef = pd.read_csv("./smap_site_backreef.csv")

    simplex_global_backreef = pd.read_csv("./simplex_global_backreef.csv")
    smap_global_backreef = pd.read_csv("./smap_global_backreef.csv")

    # Load fringing data
    simplex_local_fringing = pd.read_csv("./simplex_local_fringing.csv")
    smap_local_fringing = pd.read_csv("./smap_local_fringing.csv")

    simplex_transect_fringing = pd.read_csv("./simplex_transect_fringing.csv")
    smap_transect_fringing = pd.read_csv("./smap_transect_fringing.csv")

    simplex_site_fringing = pd.read_csv("./simplex_site_fringing.csv")
    smap_site_fringing = pd.read_csv("./smap_site_fringing.csv")

    simplex_global_fringing = pd.read_csv("./simplex_global_fringing.csv")
    smap_global_fringing = pd.read_csv("./smap_global_fringing.csv")

    # Load combined data
    simplex_local_combined = pd.read_csv("./simplex_local_combined.csv")
    smap_local_combined = pd.read_csv("./smap_local_combined.csv")

    simplex_transect_combined = pd.read_csv("./simplex_transect_combined.csv")
    smap_transect_combined = pd.read_csv("./smap_transect_combined.csv")

    simplex_site_combined = pd.read_csv("./simplex_site_combined.csv")
    smap_site_combined = pd.read_csv("./smap_site_combined.csv")

    simplex_global_combined = pd.read_csv("./simplex_global_combined.csv")
    smap_global_combined = pd.read_csv("./smap_global_combined.csv")

    # Change directory
    os.chdir('../..')
    os.chdir('results')
    os.chdir('figures')

    # Make boxplots
    # make_smap_plots(smap_local, smap_transect, smap_site, smap_global)
    # make_backreef_boxplots([smap_local_backreef, smap_transect_backreef, smap_site_backreef, smap_global_backreef])
    # make_fringing_boxplots([smap_local_fringing, smap_transect_fringing, smap_site_fringing, smap_global_fringing])

    backreef = [smap_transect_backreef, smap_site_backreef, smap_global_backreef]
    fringing = [smap_transect_fringing, smap_site_fringing, smap_global_fringing]
    combined = [smap_transect_combined, smap_site_combined, smap_global_combined]
    make_combined_boxplots(backreef, fringing, combined)







