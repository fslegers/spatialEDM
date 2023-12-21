from functions import *
import copy


class Point:
    def __init__(self, value, time_stamp, species, location):
        self.value = value
        self.time_stamp = time_stamp
        self.species = species
        self.location = location

    def display_info(self):
        print(f"Value: {self.value}")
        print(f"Observation time: {self.time_stamp}")
        print(f"Species: {self.species}")
        print(f"Location: {self.location}")


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


class TimeSeries:

    def __init__(self, points, location="loc", species='spec', interval=1):
        self.points = points
        self.location = location
        self.species = species
        self.interval = interval
        self.length = len(points)
        self.obs_times = None
        self.time_span = None
        self.start_point = None

        self.order()

    def display_info(self):
        print(f"Number of points: {self.length}")
        print(f"From location: {self.location}")
        print(f"Of species: {self.species}")
        print(f"Between times {self.time_span[0]} and {self.time_span[1]}")
        print(f"With sampling interval {self.interval}.")

    def order(self):
        sorted_points = sorted(self.points, key=lambda p: p.time_stamp)
        self.points = sorted_points
        self.obs_times = [point.time_stamp for point in self.points]
        self.time_span = [min(self.obs_times), max(self.obs_times)]
        self.start_point = sorted_points[0]

    def differentiate(self):

        new_points = []

        for i in range(1, len(self.points)):
            new_point = copy.copy(self.points[i])
            new_point.value = self.points[i].value - self.points[i - 1].value
            new_points.append(new_point)

        self.points = new_points

    def reverse_differentiate(self):

        new_points = []

        prev = self.start_point
        new_points.append(prev)

        for i in range(0, self.length - 1): # or self.length?
            new_point = copy.copy(self.points[i])
            new_point.value = self.points[i].value + prev.value
            new_points.append(new_point)
            prev = new_point

        self.points = new_points


class ConcatenatedTimeSeries:
    def __init__(self, lib: [TimeSeries]):
        self.lib = lib
        self.locations = [ts.location for ts in lib]
        self.species = [ts.species for ts in lib]
        self.interval = lib[0].interval
        self.obs_times = set([ts.obs_times for ts in lib])
        self.time_span = [min(self.obs_times), max(self.obs_times)]

    def display_info(self):
        print(f"Total number of points: {sum([ts.length for ts in self.lib])}")
        print(f"From {len(self.locations)} locations: {self.locations}")
        print(f"Of {len(self.species)} species: {self.species}")
        print(f"Between times {self.time_span[0]} and {self.time_span[1]}")
        print(f"With sampling interval {self.interval}.")

    def order(self):
        for ts in self.lib:
            ts.order()

    def differentiate(self):
        for ts in self.lib:
            ts.differentiate()

    def reverse_differentiate(self):
        for ts in self.lib:
            ts.reverse_differentiate()


class EDM():

    def __init__(self, lag=1, horizon=1, cv_method="LB", cv_fraction=0.5):
        self.cv_method = cv_method
        self.cv_fraction = cv_fraction
        self.horizon = horizon
        self.lag = lag
        self.dim = None
        self.theta = None
        self.results_simplex = {}
        self.results_smap = {}

    def correct_max_dim(self, length, max_dim):
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

    def update_results(self, result, method):

        if method == "simplex":
            self.results_simplex['corr_list'].append(result['corr'])
            self.results_simplex['rmse_list'].append(result['rmse'])
            self.results_simplex['mae_list'].append(result['mae'])

            if result['corr'] > self.results_simplex['corr'] or self.results_simplex['corr'] == -math.inf:
                self.results_simplex['observed'] = result['observed']
                self.results_simplex['predicted'] = result['predicted']
                self.results_simplex['corr'] = result['corr']
                self.results_simplex['dim'] = result['dim']
                self.dim = result['dim']


        elif method == "smap":
            self.results_smap['corr_list'].append(result['corr'])
            self.results_smap['rmse_list'].append(result['rmse'])
            self.results_smap['mae_list'].append(result['mae'])

            if result['corr'] > self.results_smap['corr'] or self.results_smap['corr'] == -math.inf:
                self.results_smap['observed'] = result['observed']
                self.results_smap['predicted'] = result['predicted']
                self.results_smap['corr'] = result['corr']
                self.results_smap['theta'] = result['theta']
                self.theta = result['theta']

        else:
            pass

    def knn_forecasting(self, X_trains, y_trains, X_tests, y_tests,
                        dim):

        result = initialize_single_result(param="dim", value=dim)

        for i in range(len(X_trains)):

            X_train, y_train, X_test, y_test = X_trains[i], y_trains[i], X_tests[i], y_tests[i]
            dist_matrix = create_distance_matrix(X_test, X_train)

            observed = []
            predicted = []

            for target in range(len(X_test)):

                dist_to_target = dist_matrix[target, :]

                if len(dist_to_target) == dim + 1:
                    nearest_neighbors = np.arange(0, dim + 1)
                else:
                    nearest_neighbors = np.argpartition(dist_to_target, min(dim + 1, len(dist_to_target) - 1))[:dim + 1]

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
                n_mae, n_corr = len(X_trains), len(X_trains)
                result, n_mae, n_corr = update_single_result(result, n_mae, n_corr, observed, predicted)

            result = average_result(result, n_mae, n_corr)

        return result

    def smap_forecasting(self, X_trains, y_trains, X_tests, y_tests, theta):

        result = initialize_single_result(param='theta', value=theta)

        for i in range(len(X_trains)):

            X_train, y_train, X_test, y_test = X_trains[i], y_trains[i], X_tests[i], y_tests[i]
            dist_matrix = create_distance_matrix(X_test, X_train)

            observed = []
            predicted = []

            for target in range(len(X_test)):
                # Calculate weights for each training point
                distances = dist_matrix[target, :]
                weights = np.exp(-theta * distances / mean(distances))

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

            n_mae, n_corr = len(X_trains), len(X_trains)
            result, n_mae, n_corr = update_single_result(result, n_mae, n_corr, observed, predicted)

        result = average_result(result, n_mae, n_corr)

        return result

    def simplex(self, ts, max_dim=10):

        self.initialize_results()
        max_dim = self.correct_max_dim(ts.length, max_dim)

        for dim in range(1, max_dim + 1):
            library = embed_time_series(ts, self.lag, dim, self.horizon)
            X_train, y_train, X_test, y_test = split_library(library, self.cv_method, self.cv_fraction)
            result = self.knn_forecasting(X_train, y_train, X_test, y_test, dim)
            self.update_results(result, "simplex")

    def smap(self, ts):

        library = embed_time_series(ts, self.lag, self.dim, self.horizon)
        X_train, y_train, X_test, y_test = split_library(library, self.cv_method, self.cv_fraction)

        for theta in range(0, 11):
            result = self.smap_forecasting(X_train, y_train, X_test, y_test, theta)
            self.update_results(result, "smap")

    def train(self, ts: TimeSeries, max_dim, plotting=True):

        if type(ts) == TimeSeries:
            ts = ConcatenatedTimeSeries([ts])

        if type(ts) != ConcatenatedTimeSeries:
            print("Input ts should either be a TimeSeries or a ConcatenatedTimeSeries")
            return 0

        self.simplex(ts, max_dim)
        self.plot_results("simplex")

        self.smap(ts)
        self.plot_results("smap")

    def predict(self, ts):
        pass
        # if type(ts) == list:
        #     if type(ts[0]) == Point:
        #         pass
        #     elif type(ts[0]) == EmbeddingVector:
        #         pass
        #     else:
        #         print('something went wrong.')
        # elif type(ts) == Point:
        #     pass
        # elif type(ts) == EmbeddingVector:
        #     pass
        # else:
        #     print('something went wrong')
        #
        # return 0

    def plot_results(self, method):

        if method == "simplex":
            results = self.results_simplex
            parameter = "dim"

        elif method == "smap":
            results = self.results_smap
            parameter = "theta"

        else:
            return 0

        # Performance measures per E or theta
        fig, axs = plt.subplots(3, sharex=True)
        fig.suptitle(method + "\n Performance measures")

        x = np.arange(1, len(results['corr_list']) + 1)

        axs[0].plot(x, results['corr_list'])
        axs[0].scatter(x, results['corr_list'])
        axs[0].set_ylabel("rho")

        axs[1].plot(x, results['mae_list'])
        axs[1].scatter(x, results['mae_list'])
        axs[1].set_ylabel("MAE")

        axs[2].plot(x, results['rmse_list'])
        axs[2].scatter(x, results['rmse_list'])
        axs[2].set_ylabel("RMSE")

        for i in range(1, len(results['corr_list']) + 1):
            axs[0].axvline(x=i, linestyle='--', color='grey', alpha=0.4)
            axs[1].axvline(x=i, linestyle='--', color='grey', alpha=0.4)
            axs[2].axvline(x=i, linestyle='--', color='grey', alpha=0.4)

        if method == "simplex":
            fig.supxlabel("dimension")
        else:
            fig.supxlabel("theta")

        axs[2].xaxis.set_major_locator(MaxNLocator(integer=True))

        fig.show()

        # Observed vs predictions
        fig2, axs2 = plt.subplots()

        axs2.scatter(results['observed'], results['predicted'])
        min_ = min([min(results['observed']), min(results['predicted'])])
        max_ = max([max(results['observed']), max(results['predicted'])])
        axs2.plot([min_, max_], [min_, max_])
        fig2.suptitle(method + "\n Observed vs Predicted")
        axs2.set_xlabel("Observed")
        axs2.set_ylabel("Predicted")
        # plt.show()

        fig2.show()


class GPR():
    pass


class HierarchicalGPR():
    pass





if __name__ == "__main__":
    point_1 = Point(1, 1, "Aap", "Helmond")
    point_2 = Point(2, 2, "Aap", "Utrecht")
    point_3 = Point(3, 3, "Koe", "Amsterdam")

    point_1.display_info()

    time_series = TimeSeries([point_1, point_2, point_3])

    time_series.display_info()

    print("hee")
