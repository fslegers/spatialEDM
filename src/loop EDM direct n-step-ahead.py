from EDM import *
from GPR import *
from functools import partial
import multiprocessing as mp
import pandas as pd
import os
from tqdm import tqdm

def inner_loop(train, test, method, cv, sd, n_iter, horizon):

    path_name = "rho=28 EDM LB_CV 250sim long_test_set"

    interval_list = [1,5,10,20]
    n_points_list = [15,20,25,30,40,50,75,100,150,300]

    performance_training = np.zeros((len(interval_list), len(n_points_list)))
    performance_test = np.zeros((len(interval_list), len(n_points_list)))

    for i in range(len(interval_list)):
        for j in range(len(n_points_list)):

            # Take sample(s) from lorenz trajectory
            x_train, t = sample_from_ts(train, np.arange(0, len(train)),
                                      n_points=n_points_list[j],
                                      sampling_interval=interval_list[i],
                                      sample_end=True)

            x_test, t = sample_from_ts(test, np.arange(0, len(test)),
                                       sampling_interval=interval_list[i])

            # Start training the model
            results = edm(x_train, lag=1, max_E=10, plotting=False, CV=cv)

            # Save performance on training data
            performance_training[i, j] = round(results['corr'],5)

            # Embed data with optimal E
            lib_train = embed_time_series(x_train, lag=1, E=results['E'])
            lib_test = embed_time_series(x_test, lag=1, E=results['E'])

            # Split target value
            X_train, y_train = [], []
            for point in lib_train:
                X_train.append(point[0])
                y_train.append(point[1])

            X_test, y_test = [], []
            for point in lib_test:
                X_test.append(point[0])
                y_test.append(point[1])

            #TODO: turn into recursive n-step-ahead
            # Make predictions on the test set
            results = smap_forecasting([X_train], [y_train], [X_test], [y_test], results['theta'])


            # Save performance on test data
            performance_test[i, j] = round(results['corr'],5)

    # Save performance matrices on computer
    path = "C:/Users/5605407/Documents/PhD/PythonProjects/timeseriesanalysis/results/output/" + path_name + "/sd = " + str(sd)

    if not os.path.exists(path):
        os.makedirs(path)

    performance_test = pd.DataFrame(performance_test)
    performance_training = pd.DataFrame(performance_training)

    performance_test.to_csv(path + "./" + str(n_iter) + ", test.csv")
    performance_training.to_csv(path + "./" + str(n_iter) + ", training.csv")

    return 0


def outer_loop(sd, ts, n_traj, shift, test_size, method, cv_method, horizon):

    sd_list=[0,0.05,0.1,0.5,1,2.5,5,10]

    # Sample multiple trajectories from ts
    loop = tqdm(total=n_traj, desc="process " + str(sd_list.index(sd)), leave=False)

    for i in range(n_traj):
        start_ = int(i * shift)
        stop_ = int(start_ + 350*30 + test_size)

        # Take a slice from the long time series
        x = ts[start_:stop_]

        # Add noise
        noise = np.random.normal(0, sd, len(x))
        x_with_noise = x + noise

        # Split into training and test set
        x_train = x_with_noise[:-test_size]
        x_test = x_with_noise[-test_size:]

        # Perform training and testing and save results
        inner_loop(x_train, x_test, method, cv_method, sd, i, horizon)
        loop.update(1)

    return 0

def run_simulation(method="EDM", cv="LB", horizon=0.1):

    n_chunks_of_trajectory = 250
    resolution_lorenz = 0.05
    H = int(np.ceil(horizon/resolution_lorenz))
    print(str(H) +"-step ahead predictions")
    t_between_chuncks = 1
    length_test_set = 10

    initial_point = [4.548120346844322, -2.081443690988742, 30.804556029728243]
    t_max = int(np.ceil((n_chunks_of_trajectory*t_between_chuncks + 1) + 17500*resolution_lorenz))
    n_time_steps = int(np.ceil(t_max/resolution_lorenz))

    # Simulate one giant trajectory of the Lorenz system
    x, y, z, t = simulate_lorenz(initial_point, [10, 28, 8/3], n_time_steps, t_max, 0)
    del(y,z,t)

    sd_list = [0,0.05,0.1,0.5,1,2.5,5,10]
    #sd_list = [0]

    # Start parallel loop
    pool = mp.Pool(mp.cpu_count()-2)
    pool.map(partial(outer_loop,
                     ts=x,
                     n_traj=n_chunks_of_trajectory,
                     shift=int(t_between_chuncks/resolution_lorenz),
                     test_size=int(length_test_set/resolution_lorenz),
                     method=method,
                     cv_method=cv,
                     horizon=H), sd_list)

    pool.close()
    pool.join()


if __name__ == "__main__":

    run_simulation(cv="LB")