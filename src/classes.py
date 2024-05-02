import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning
import statistics

from sklearn.linear_model import LinearRegression


class Point:
    def __init__(self, value, time_stamp, species, location):
        self.value = value
        self.time_stamp = time_stamp
        self.species = species
        self.loc = location

    def display_info(self):
        print(f"Value: {self.value}")
        print(f"Observation time: {self.time_stamp}")
        print(f"Species: {self.species}")
        print(f"Location: {self.loc}")


class EmbeddingVector:
    def __init__(self, values, next_time_stamp, species, location):
        self.values = values
        self.next_time_stamp = next_time_stamp
        self.species = species
        self.location = location

    def display_info(self):
        print(f"Values: {self.values}")
        print(f"Observation time: {self.time_stamp}")
        print(f"Species: {self.species}")
        print(f"Location: {self.location}")


class Library:
    def __init__(self, points, dim, lag, horizon, cv_method, cv_fraction):
        self.points = points
        self.dim = dim
        self.lag = lag
        self.horizon = horizon
        self.cv_method = cv_method
        self.cv_fraction = cv_fraction
        self.train = []
        self.test = []

        self.fill_libraries()

    def fill_libraries(self):

        locations = set([point.loc for point in self.points])
        species = set([point.species for point in self.points])

        for loc in locations:
            for spec in species:

                lib = self.embed_time_series(loc, spec)
                train, test = self.split_library(lib)

                self.train += train
                self.test += test

    def create_hankel_matrix(self, ts):
        """
        Returns the first E+1 rows of the Hankel-matrix of a time series. Each consecutive row contains
        the time series shifted backwards lag time steps. Should only contain one location,species pair!
        """
        # TODO: integrate time stamps into this

        dim = self.dim
        lag = self.lag
        hankel_matrix = []

        for i in range(dim + 1):
            if i == 0:
                delayed_ts = ts[(dim - i) * lag:]  # Add original time series
            else:
                delayed_ts = ts[(dim - i) * lag:-i * lag]  # Add time series that is shifted i times
            hankel_matrix.append(delayed_ts)

        hankel_matrix = np.stack(hankel_matrix, axis=0)  # turn list into np.array

        return hankel_matrix

    def shift_hankel_matrix(self, matrix):
        """
        Shift first row to the left horizon-1 times
        """
        if self.horizon > 1:
            one_step_ahead = matrix[0, :]
            n_step_ahead = one_step_ahead[self.horizon - 1:]
            matrix = matrix[:, :-(self.horizon - 1)]
            matrix[0, :] = n_step_ahead

        return matrix

    def hankel_to_lib(self, matrix):
        """From a hankel matrix, creates a library of input,output-pairs"""
        lib = []
        for col in range(matrix.shape[1]):
            t, spec, loc = matrix[0, col].time_stamp, matrix[0, col].species, matrix[0, col].loc
            x = EmbeddingVector([point.value for point in matrix[1:, col]], t, spec, loc)
            y = matrix[0, col]
            lib.append([x, y])
        return lib

    def embed_time_series(self, location, species):

        # Select subset of points and turn into TimeSeries
        points = [point for point in self.points if point.loc == location and point.species == species]

        # Create hankel_matrix and shift if horizon != 1
        hankel_matrix = self.create_hankel_matrix(points)
        hankel_matrix = self.shift_hankel_matrix(hankel_matrix)

        # From hankel_matrix, save pairs in lib
        lib = self.hankel_to_lib(hankel_matrix)

        return lib

    def split_library(self, lib):
        if self.cv_method == "LB":
            train, test = self.split_last_block(lib)

        elif self.cv_method == "RB":
            train, test = self.split_rolling_base(lib)

        return train, test

    def split_last_block(self, lib):

        # Split predictor from response variables
        X, y = [], []
        t_min, t_max = math.inf, -math.inf

        for point in lib:
            X.append(point[0])
            y.append(point[1])

            if point[1].time_stamp < t_min:
                t_min = point[1].time_stamp
            if point[1].time_stamp > t_max:
                t_max = point[1].time_stamp

        # Split into training and test set (time ordered)
        cut_off = int(math.ceil(t_min + (t_max - t_min) * self.cv_fraction))  # TODO: naar boven of onder afronden?

        X_train, y_train, X_test, y_test = [], [], [], []
        for i in range(len(X)):
            if y[i].time_stamp <= cut_off:
                X_train.append(X[i])
                y_train.append(y[i])
            else:
                X_test.append(X[i])
                y_test.append(y[i])

        train = list(zip(X_train, y_train))
        test = list(zip(X_test, y_test))

        return train, test

    def split_rolling_base(self, lib): # TODO: test

        train, test = [], []

        X, y = [], []
        t_min, t_max = math.inf, -math.inf
        for point in lib:
            X.append(point[0])
            y.append(point[1])

            if point[1].time_stamp < t_min:
                t_min = point[1].time_stamp
            if point[1].time_stamp > t_max:
                t_max = point[1].time_stamp

        for run in range(int(1/self.cv_fraction)):

            # Split into training and test set (time ordered)
            upper_bound = min(t_max, int(math.ceil(t_max - run * self.cv_fraction * (t_max - t_min))))
            lower_bound = max(t_min, int(math.ceil(t_max - (run + 1) * self.cv_fraction * (t_max - t_min))))

            X_train, y_train, X_test, y_test = [], [], [], []
            for i in range(len(X)):
                if y[i].time_stamp > lower_bound and y[i].time_stamp <= upper_bound:
                    X_train.append(X[i])
                    y_train.append(y[i])
                else:
                    X_test.append(X[i])
                    y_test.append(y[i])

            train_this_run = list(zip(X_train, y_train))
            test_this_run = list(zip(X_test, y_test))

            train.append(train_this_run)
            test.append(test_this_run)

        return train, test


class EDM():

    def __init__(self, lag=1, horizon=1, max_dim=10, cv_method="LB", cv_fraction=0.5):
        self.lag = lag
        self.dim = None
        self.max_dim = max_dim
        self.theta = None
        self.horizon = horizon
        self.cv_method = cv_method
        self.cv_fraction = cv_fraction
        self.results_simplex = {}
        self.results_smap = {}
        self.lib = []
        self.training_ts = []
        self.interval = 0

        self.initialize_results()


    def correct_max_dim(self, points, max_dim):

        length = math.inf

        locations = set([point.loc for point in points])

        for loc in locations:
            loc_length = len([point for point in points if point.loc == loc])
            if  loc_length < length:
                length = loc_length

        if max_dim > (length - 2 - self.horizon) / self.lag:
            max_dim = max(1, (length - 2 - 1) / self.horizon)

        return max_dim


    def initialize_results(self):
        self.results_simplex['corr'] = -math.inf
        self.results_simplex['mae'] = math.inf
        self.results_simplex['rmse'] = math.inf
        self.results_simplex['corr_list'] = []
        self.results_simplex['mae_list'] = []
        self.results_simplex['rmse_list'] = []
        self.results_simplex['dim'] = None

        self.results_smap['corr'] = -math.inf
        self.results_smap['mae'] = math.inf
        self.results_smap['rmse'] = math.inf
        self.results_smap['corr_list'] = []
        self.results_smap['mae_list'] = []
        self.results_smap['rmse_list'] = []
        self.results_smap['theta'] = None


    def get_single_result(self, obs, pred):

        diff = np.subtract(obs, pred)
        result = {}

        result['mae'] = np.mean(abs(diff))
        result['rmse'] = math.sqrt(np.mean(np.square(diff)))
        result['observed'] = obs
        result['predicted'] = pred

        try:
            result['corr'] = pearsonr(obs, pred)[0]
        except ConstantInputWarning:
            result['corr'] = None
        except:
            result['corr'] = None

        return result


    def update_results(self, result, var, method):

        if method == "simplex":
            self.results_simplex['corr_list'].append(result['corr'])
            self.results_simplex['rmse_list'].append(result['rmse'])
            self.results_simplex['mae_list'].append(result['mae'])

            if result['rmse'] < self.results_simplex['rmse'] or self.results_simplex['rmse'] == math.inf:
                self.results_simplex['observed'] = result['observed']
                self.results_simplex['predicted'] = result['predicted']
                self.results_simplex['corr'] = result['corr']
                self.results_simplex['rmse'] = result['rmse']
                self.results_simplex['mae'] = result['mae']
                self.results_simplex['dim'] = var
                self.dim = var

        elif method == "smap":
            self.results_smap['corr_list'].append(result['corr'])
            self.results_smap['rmse_list'].append(result['rmse'])
            self.results_smap['mae_list'].append(result['mae'])

            if result['rmse'] < self.results_smap['rmse'] or self.results_smap['rmse'] == math.inf:
                self.results_smap['observed'] = result['observed']
                self.results_smap['predicted'] = result['predicted']
                self.results_smap['corr'] = result['corr']
                self.results_smap['rmse'] = result['rmse']
                self.results_smap['mae'] = result['mae']
                self.results_smap['theta'] = var
                self.theta = var
        else:
            print("Something went wrong (line 307).")


    def knn_forecasting(self, lib):

        if lib.cv_method == "LB":
            X_train, y_train = list(zip(*lib.train))
            X_test, y_test = list(zip(*lib.test))

            dist_matrix = create_distance_matrix(X_test, X_train)

            observed = []
            predicted = []

            for target in range(len(X_test)):

                dist_to_target = dist_matrix[target, :]

                if len(dist_to_target) == lib.dim + 1:
                    nearest_neighbors = np.arange(0, lib.dim + 1)
                else:
                    nearest_neighbors = np.argpartition(dist_to_target, min(lib.dim + 1, len(dist_to_target) - 1))[:lib.dim + 1]

                min_distance = dist_to_target[nearest_neighbors[0]]

                weighted_average = 0
                total_weight = 0
                weights = []

                # Calculate weighted sum of next value
                for neighbor in nearest_neighbors:
                    if min_distance == 0:
                        if dist_to_target[neighbor] == 0:
                            weight = 1
                        else:
                            weight = 0.000001
                    else:
                        weight = np.exp(-dist_to_target[neighbor] / min_distance)

                    next_val = y_train[neighbor].value
                    weighted_average += next_val * weight
                    total_weight += weight
                    weights.append(weight)

                # Calculate weighted average
                weighted_average = weighted_average / total_weight
                predicted.append(weighted_average)
                observed.append(y_test[target].value)

            result = self.get_single_result(obs=observed, pred=predicted)

        elif lib.cv_method == "RB":

            for i in range(len(lib.train)):

                train, test = lib.train[i], lib.test[i]

                X_train, y_train = list(zip(*train))
                X_test, y_test = list(zip(*test))

                dist_matrix = create_distance_matrix(X_test, X_train)

                observed = []
                predicted = []
                all_results = []

                try:
                    for target in range(len(X_test)):

                        dist_to_target = dist_matrix[target, :]
                        if len(dist_to_target) == lib.dim + 1:
                            nearest_neighbors = np.arange(0, lib.dim + 1)
                        else:
                            nearest_neighbors = np.argpartition(dist_to_target,
                                                                min(lib.dim + 1, len(dist_to_target) - 1))[:lib.dim + 1]

                        min_distance = dist_to_target[nearest_neighbors[0]]

                        weighted_average = 0
                        total_weight = 0
                        weights = []

                        # Calculate weighted sum of next value
                        for neighbor in nearest_neighbors:
                            if min_distance == 0:
                                if dist_to_target[neighbor] == 0:
                                    weight = 1
                                else:
                                    weight = 0.000001
                            else:
                                weight = np.exp(-dist_to_target[neighbor] / min_distance)

                            next_val = y_train[neighbor].value
                            weighted_average += next_val * weight
                            total_weight += weight
                            weights.append(weight)

                        # Calculate weighted average
                        weighted_average = weighted_average / total_weight
                        predicted.append(weighted_average)
                        observed.append(y_test[target].value)

                    # Update result
                    result = self.get_single_result(observed, predicted)
                    all_results.append(result)

                except:
                    print("Something went wrong (line 493)")

            result = average_result(all_results)

        return result


    def smap_forecasting(self, lib, theta):

        if lib.cv_method == "LB":
            X_train, y_train = list(zip(*lib.train))
            X_test, y_test = list(zip(*lib.test))

            dist_matrix = create_distance_matrix(X_test, X_train)

            observed = []
            predicted = []

            for target in range(len(X_test)):
                # Calculate weights for each training point
                distances = dist_matrix[target, :]
                weights = np.exp(-theta * distances / np.mean(distances))

                # Fill vector of weighted future values of training set
                next_values = [point.value for point in y_train]
                b = np.multiply(weights, next_values)

                # Fill matrix A
                embedding_vectors = np.array([point.values for point in X_train])
                A = embedding_vectors * np.array(weights)[:, None]

                # Calculate coefficients C using the pseudo-inverse of A (via SVD)
                coeffs = np.matmul(np.linalg.pinv(A), b)

                # Make prediction
                prediction = np.matmul(np.array(X_test[target].values), coeffs)
                observed.append(y_test[target].value)
                predicted.append(prediction)

            result = self.get_single_result(obs=observed, pred=predicted)

        elif lib.cv_method == "RB":

            for i in range(len(lib.train)):

                train, test = lib.train[i], lib.test[i]

                X_train, y_train = list(zip(*train))
                X_test, y_test = list(zip(*test))

                dist_matrix = create_distance_matrix(X_test, X_train)

                observed = []
                predicted = []

                all_results = []

                try:
                    for target in range(len(X_test)):
                        # Calculate weights for each training point
                        distances = dist_matrix[target, :]
                        weights = np.exp(-theta * distances / np.mean(distances))

                        # Fill vector of weighted future values of training set
                        next_values = [point.value for point in y_train]
                        b = np.multiply(weights, next_values)

                        # Fill matrix A
                        embedding_vectors = np.array([point.values for point in X_train])
                        A = embedding_vectors * np.array(weights)[:, None]

                        # Calculate coefficients C using the pseudo-inverse of A (via SVD)
                        coeffs = np.matmul(np.linalg.pinv(A), b)

                        # Make prediction
                        prediction = np.matmul(np.array(X_test[target].values), coeffs)
                        observed.append(y_test[target].value)
                        predicted.append(prediction)

                    result = self.get_single_result(obs=observed, pred=predicted)
                    all_results.append(result)

                except:
                    print("Something went wrong (line 493)")

            result = average_result(all_results)

        return result


    def simplex(self, points, max_dim=10):

        self.initialize_results()
        max_dim = self.correct_max_dim(points, max_dim)

        for dim in range(1, max_dim + 1):
            lib = Library(points, dim, self.lag, self.horizon, self.cv_method, self.cv_fraction)
            result = self.knn_forecasting(lib)
            self.update_results(result, dim, "simplex")


    def smap(self, points):

        lib = Library(points, self.dim, self.lag, self.horizon, self.cv_method, self.cv_fraction)

        for theta in range(0, 11):
            result = self.smap_forecasting(lib, theta)
            self.update_results(result, theta, "smap")


    def train(self, ts, interval = 1, max_dim=10):
        """
        Performs training of EDM. First performs Simplex, then S-Map.
        :param ts: one list of points, may come from multiple time series. Must be one list and all elements are Point.
        """

        #print("Start training")
        self.interval = interval

        #print("Simplex..........")
        self.simplex(ts, max_dim)

        #print("S-Map............")
        self.smap(ts)

        #self.plot_results()
        self.lib = Library(ts, self.dim, self.lag, self.horizon, "LB", 1).train
        self.training_ts = ts


    def embed_test_data(self, ts):
        embedding_vectors = []

        # Create embedding vectors for the test data
        locations = set([point.loc for point in ts])
        species = set([point.species for point in ts])

        for loc in locations:
            for spec in species:
                points = [point for point in ts if point.loc == loc and point.species == spec]

                # Add from training library (if possible), the dim previous values of the test points
                min_ = min([point.time_stamp for point in points])
                selected_points = [point for point in self.training_ts if point.loc == loc and point.species == spec
                                   and point.time_stamp >= min_ - self.dim*self.interval and point.time_stamp < min_]
                points = selected_points + points
                points = sorted(points, key=lambda p: p.time_stamp)

                # Create hankel_matrix
                hankel_matrix = []

                for i in range(self.dim):   #TODO: Check if changing from self.dim + 1 to this indeed removed the x_t
                    if i == 0:
                        delayed_ts = points[(self.dim - i) * self.lag:]
                    else:
                        delayed_ts = points[(self.dim - i) * self.lag:-i * self.lag]
                    hankel_matrix.append(delayed_ts)

                # Turn hankel_matrix into EmbeddingVector
                hankel_matrix = np.array(hankel_matrix)

                for j in range(hankel_matrix.shape[1]):
                    values = [hankel_matrix[i, j].value for i in range(hankel_matrix.shape[0])]
                    embedding_vec = EmbeddingVector(values, hankel_matrix[0,j].time_stamp + self.interval, spec, loc)
                    embedding_vectors.append(embedding_vec)

        return embedding_vectors


    def predict_new_points_simplex(self, X_test):

        X_train, y_train = map(list, zip(*self.lib))

        dist_matrix = create_distance_matrix(X_test, X_train)
        predicted = []

        for target in range(len(X_test)):

            dist_to_target = dist_matrix[target, :]

            if len(dist_to_target) == self.dim + 1:
                nearest_neighbors = np.arange(0, self.dim + 1)
            else:
                nearest_neighbors = np.argpartition(dist_to_target, min(self.dim, len(dist_to_target) - 1))[
                                    :self.dim + 1]

            min_distance = dist_to_target[nearest_neighbors[0]]

            weighted_average = 0
            total_weight = 0
            weights = []

            # Calculate weighted sum of next value
            for neighbor in nearest_neighbors:
                if min_distance == 0:
                    if dist_to_target[neighbor] == 0:
                        weight = 1
                    else:
                        weight = 0.000001
                else:
                    weight = np.exp(-dist_to_target[neighbor] / min_distance)

                next_val = y_train[neighbor].value
                weighted_average += next_val * weight
                total_weight += weight
                weights.append(weight)

            # Calculate weighted average
            weighted_average = weighted_average / total_weight

            prediction = Point(weighted_average, X_test[target].next_time_stamp, X_test[target].species, X_test[target].location)
            predicted.append(prediction)

        return predicted


    def predict_new_points_smap(self, X_test):
        X_train, y_train = map(list, zip(*self.lib))
        dist_matrix = create_distance_matrix(X_test, X_train)
        predicted = []

        for target in range(len(X_test)):
            # Calculate weights for each training point
            distances = dist_matrix[target, :]
            weights = np.exp(-self.theta * distances / np.mean(distances))

            # Fill vector of weighted future values of training set
            next_values = [point.value for point in y_train]
            b = np.multiply(weights, next_values)

            # Fill matrix A
            embedding_vectors = np.array([point.values for point in X_train])
            A = embedding_vectors * np.array(weights)[:, None]

            # Calculate coefficients C using the pseudo-inverse of A (via SVD)
            coeffs = np.matmul(np.linalg.pinv(A), b)

            # Make prediction
            prediction = np.matmul(np.array(X_test[target].values), coeffs)

            prediction = Point(prediction, X_test[target].next_time_stamp, X_test[target].species,
                               X_test[target].location)
            predicted.append(prediction)

        return predicted


    def predict(self, ts, hor=1):

        X_test = self.embed_test_data(ts)

        results_simplex = []
        results_smap = []

        for i in range(hor):
            # Simplex
            pred_simplex = self.predict_new_points_simplex(X_test)
            results_simplex.append(pred_simplex)

            # S-Map
            pred_smap = self.predict_new_points_smap(X_test)
            results_smap.append(pred_smap)

            # Update the embedding vectors X_test
            for pair in X_test:

                # discard oldest value
                pair.values = pair.values[:-1]

                # add prediction
                new_value = [point.value for point in pred_smap if point.time_stamp == pair.next_time_stamp][0]
                pair.values.insert(0, new_value)

                # update next_time_stamp
                pair.next_time_stamp += 1

        ### Save results with correct horizon ###
        # Get observed values
        df_obs = {'location': [point.loc for point in ts],
                  'species': [point.species for point in ts],
                  'time_stamp': [point.time_stamp for point in ts],
                  'obs': [point.value for point in ts]}
        df_obs = pd.DataFrame(df_obs)

        # Get predictions + horizon for Simplex
        df_pred = pd.DataFrame()
        for hor, sublist in enumerate(results_simplex, start=1):
            df_sub =pd.DataFrame({'location': [point.loc for point in results_simplex[hor-1]],
                       'species': [point.species for point in results_simplex[hor-1]],
                       'time_stamp': [point.time_stamp for point in results_simplex[hor-1]],
                       'pred': [point.value for point in results_simplex[hor-1]]})
            df_sub['hor'] = hor
            df_pred = pd.concat([df_pred, df_sub], axis=0)

        # Merge observations and predictions for Simplex
        df_simplex = pd.merge(df_pred, df_obs, on=['location', 'species', 'time_stamp'], how='left')

        # Get predictions + horizon for S-Map
        df_pred = pd.DataFrame()
        for hor, sublist in enumerate(results_smap, start=1):
            df_sub = pd.DataFrame({'location': [point.loc for point in results_smap[hor-1]],
                       'species': [point.species for point in results_smap[hor-1]],
                       'time_stamp': [point.time_stamp for point in results_smap[hor-1]],
                       'pred': [point.value for point in results_smap[hor-1]]})
            df_sub['hor'] = hor
            df_pred = pd.concat([df_pred, df_sub], ignore_index=True)

        # Merge observations and predictions for S-Map
        df_smap = pd.merge(df_pred, df_obs, on=['location', 'species', 'time_stamp'], how='left')

        return df_simplex, df_smap

    def plot_results(self):

        plt.style.use('bmh')

        results = self.results_simplex

        # Performance measures per E or theta
        fig, axs = plt.subplots(2, 2, figsize = (10,8))

        # Add titles
        fig.text(0.25, 0.92, 'Simplex', ha='center', va='center', fontsize=16)
        fig.text(0.75, 0.92, 'S-Map', ha='center', va='center', fontsize=16)

        x = np.arange(1, len(results['corr_list']) + 1)

        axs[0,0].plot(x, results['corr_list'])
        axs[0,0].scatter(x, results['corr_list'])
        axs[0,0].set_ylabel("rho")

        # TODO: make plots for RMSE/MAE

        axs[0,0].set_xlabel("dimension")

        # Observed vs predictions

        axs[1,0].scatter(results['observed'], results['predicted'])
        min_ = min([min(results['observed']), min(results['predicted'])])
        max_ = max([max(results['observed']), max(results['predicted'])])
        axs[1,0].plot([min_, max_], [min_, max_])
        #axs[1,0].set_title("Simplex\n Observed vs Predicted")
        axs[1,0].set_xlabel("Observed")
        axs[1,0].set_ylabel("Predicted")

        ### SMAP ###
        results = self.results_smap
        x = np.arange(1, len(results['corr_list']) + 1)

        axs[0,1].plot(x, results['corr_list'])
        axs[0,1].scatter(x, results['corr_list'])
        axs[0,1].set_ylabel("rho")

        axs[0,1].set_xlabel("theta")
        axs[0,1].sharey(axs[0,0])

        # Observed vs predictions

        axs[1,1].scatter(results['observed'], results['predicted'])
        min_ = min([min(results['observed']), min(results['predicted'])])
        max_ = max([max(results['observed']), max(results['predicted'])])
        axs[1,1].plot([min_, max_], [min_, max_])
        axs[1,1].set_xlabel("Observed")
        axs[1,1].set_ylabel("Predicted")

        line = plt.Line2D((.5, .5), (.1, .95), color="k", linewidth=1)
        fig.add_artist(line)
        plt.tight_layout(rect=[0, 0.1, 1, 0.9])

        fig.show()


    def plot_predictions(self, ts, df_simplex, df_smap, path = ""):

        #TODO: Doesn't work when horizon = 1

        plt.style.use('bmh')
        hor = max(df_simplex['hor'])
        subplots = []

        ### OBSERVED VERSUS PREDICTED ###
        min_ = min(np.minimum.reduce([df_simplex['pred'], df_simplex['obs'], df_smap['pred'], df_smap['obs']]))
        max_ = max(np.maximum.reduce([df_simplex['pred'], df_simplex['obs'], df_smap['pred'], df_smap['obs']]))

        j = 0
        for i in range(hor):
            fig_00, ax_00 = plt.subplots()
            ax_00.plot([min_, max_], [min_, max_])
            ax_00.scatter(df_simplex[df_simplex['hor'] == i + 1]['obs'],
                          df_simplex[df_simplex['hor'] == i + 1]['pred'])
            x = (df_simplex[df_simplex['hor'] == i + 1]['obs']).values
            y = (df_simplex[df_simplex['hor'] == i + 1]['pred']).values
            indices = ~np.isnan(x) & ~np.isnan(y)
            r, p = pearsonr(np.array(x)[indices], np.array(y)[indices])
            plt.text(.79, .15, 'r={:.3f}'.format(r), fontsize=11, transform=ax_00.transAxes)
            ax_00.set_aspect('equal')
            ax_00.set_xlabel('Observed')
            ax_00.set_ylabel('Predicted')
            fig_00.tight_layout()
            fig_00.savefig(f'plot_{j + 1}.png')
            j += 1
            subplots.append(fig_00)

            fig_01, ax_01 = plt.subplots()
            ax_01.plot([min_, max_], [min_, max_])
            ax_01.scatter(df_smap[df_smap['hor'] == i+1]['obs'],
                          df_smap[df_smap['hor'] == i+1]['pred'])
            x = (df_smap[df_smap['hor'] == i + 1]['obs']).tolist()
            y = (df_smap[df_smap['hor'] == i + 1]['pred']).tolist()
            indices = ~np.isnan(x) & ~np.isnan(y)
            r, p = pearsonr(np.array(x)[indices],np.array(y)[indices])
            plt.text(.79, .15, 'r={:.3f}'.format(r), fontsize=11, transform=ax_01.transAxes)
            ax_01.set_aspect('equal')
            ax_01.set_xlabel('Observed')
            ax_01.set_ylabel('Predicted')
            fig_01.tight_layout()
            fig_01.savefig(f'plot_{j + 1}.png')
            j += 1
            subplots.append(fig_01)

        # Create a big figure to display the saved subplots
        fig_big, axs_big = plt.subplots(hor, 2, figsize=(10,8))

        # Load and display the saved subplots in the big figure
        for i, subplot in enumerate(subplots):
            img = plt.imread(f'plot_{i + 1}.png')

            row = i // 2
            col = i % 2

            if len(axs_big.shape) == 1:
                axs_big[col].imshow(img)
                axs_big[col].axis('off')
                axs_big[col].text(0, 0, f'1 step ahead', fontsize=14, rotation='vertical', position=(-2, 300))

                # Add titles above each column
                axs_big[0].set_title('Simplex')
                axs_big[1].set_title('S-Map')

            else:
                # If there are multiple rows, axs_big is a 2D array
                axs_big[row, col].imshow(img)
                axs_big[row, col].axis('off')

                if i % 2 == 0:
                    axs_big[row, col].text(0, 0, f'{i // 2 + 1} steps ahead', fontsize=14, rotation='vertical', position = (-2, 300))

                # Add titles above each column
                axs_big[0, 0].set_title('Simplex')
                axs_big[0, 1].set_title('S-Map')

        # Show the big figure
        plt.tight_layout()
        plt.savefig("big_plot")
        plt.show()

        return 0


def average_result(results):

    result = {}
    result['observed'], result['predicted'] = [], []

    for single_result in results:
        result['observed'].append(single_result['observed'])
        result['predicted'].append(single_result['predicted'])

    corr_values = [result['corr'] for result in results]
    mae_values = [result['mae'] for result in results]
    rmse_values = [result['rmse'] for result in results]

    try:
        result['corr'] = statistics.mean(corr_values)
    except:
        result['corr'] = None

    try:
        result['mae'] = statistics.mean(mae_values)
        result['rmse'] = statistics.mean(rmse_values)
    except:
        result['mae'] = None
        result['rmse'] = None

    return result


def transform_array_to_ts(values, time_stamps = None, spec = "spec1", loc = "loc1"):
    """
    Transforms a list of numerical values into a list of Points.
    :param values: measurements
    :param time_stamps: time stamps. Should be a list of equal size and order as values.
    :param spec: name of the species of the measurements
    :param loc: name of the location of the measurements
    :return:
    """

    if time_stamps is None:
        time_stamps = list(range(1, len(values) + 1))

    try:
        len(time_stamps) == len(values)
    except:
        print("List of time stamps is not of equal size as list of measurements.")
        time_stamps = list(range(1, len(values) + 1))

    points = []
    for i in range(len(values)):
        point = Point(values[i], time_stamps[i], spec, loc)
        points.append(point)

    return points


def create_distance_matrix(X, Y):
    """
    Returns a matrix of distances between time-delayed embedding vectors in Y to vectors in X.
    """
    dist_matrix = np.zeros((len(X), len(Y)))

    for p in range(len(X)):
        for q in range(len(Y)):
            x = np.array(X[p].values)
            y = np.array(Y[q].values)
            dist = np.linalg.norm((y - x))
            dist_matrix[p, q] = dist

    return dist_matrix


def format_y_axis(value, _):
    return f"{value:.2f}"


def preprocessing(x, t, loc):

    x, trend_info = remove_linear_trend(x, t)
    x, mean, std_dev = normalize(x)

    return x, (loc, {'trend': trend_info, 'mean': mean, 'std_dev': std_dev})


def reverse_preprocessing(result, preprocessing_info):

    new_result = []

    for loc in np.unique(result['location']):
        df_result = result[result['location'] == loc]

        for x,y in preprocessing_info:
            if x == loc:
                trend = y['trend']
                mean = y['mean']
                std_dev = y['std_dev']
                break

        new_observed = reverse_normalization(df_result['obs'], mean, std_dev)
        new_observed = add_linear_trend(new_observed, trend, df_result['time_stamp'].values)

        new_predicted = reverse_normalization(df_result['pred'], mean, std_dev)
        new_predicted = add_linear_trend(new_predicted, trend, df_result['time_stamp'].values)

        df_result = pd.DataFrame({
            'location': [loc] * len(new_observed),
            'species': ['A'] * len(new_observed),
            'time_stamp': df_result['time_stamp'],
            'obs': new_observed,
            'pred': new_predicted
        })

        new_result.append(df_result)

    return pd.concat(new_result, ignore_index=True)


def normalize(values):

    "Scale values to zero-mean and unit variance"

    mean = np.mean(values)
    std_dev = np.std(values)

    normalized_data = (values - mean)/std_dev

    return normalized_data, mean, std_dev


def reverse_normalization(values, mean, std_dev):
    "Transform values back to original range"
    data = std_dev * values + mean
    return data


def remove_linear_trend(values, time_stamps=[]):

    if not time_stamps:
        time_stamps = range(len(values))

    if not isinstance(time_stamps, np.ndarray):
        time_stamps = np.array(time_stamps)

    x = time_stamps.reshape(-1,1)
    model = LinearRegression().fit(x,values)
    trend = model.predict(x)
    detrended_data = values - trend

    return detrended_data, model


def add_linear_trend(values, model, time_stamps=[]):

    try:
        if not time_stamps:
            time_stamps = range(len(values))
        else:
            pass
    except:
        pass

    if not isinstance(time_stamps, np.ndarray):
        time_stamps = np.array(time_stamps)

    x = time_stamps.reshape(-1,1)
    trend = model.predict(x)
    values += trend

    return values


if __name__ == "__main__":

    x = np.arange(1, 35)
    y_1 = np.sin(x / 10.0)

    x = np.arange(5, 40)
    y_2 = np.sin(x / 10.0)

    ts_1 = transform_array_to_ts(y_1, loc="A")
    ts_2 = transform_array_to_ts(y_2, loc="B")

    ts_train = ts_1[:20] + ts_2[:20]
    ts_test = ts_1[19:25] + ts_2[19:25]

    model = EDM(cv_method="RB", cv_fraction=0.25)
    model.train(ts_train)

    simplex, smap = model.predict(ts_test, 1)
    model.plot_predictions(ts_test, simplex, smap)


