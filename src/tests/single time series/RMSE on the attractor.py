import time

import pandas as pd

from src.classes import *
from src.simulate_lorenz import *

from multiprocessing import Pool
from functools import partial

import numpy as np
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_lorenz(x, y, z, t):
    x_ = [x[i] for i in range(15000) if i % 5 == 0]
    y_ = [y[i] for i in range(15000) if i % 5 == 0]
    z_ = [z[i] for i in range(15000) if i % 5 == 0]
    t_ = [t[i] for i in range(15000) if i % 5 == 0]

    return x_, y_, z_, t_

def get_one_RMSE(i, x, t, length, hor):

    # Take segment from big lorenz trajectory
    x_subset, t_subset = x[i:i+length+hor], t[i:i+length+hor]

    # Transform time series into EDM library
    collection = []
    for j in np.arange(length+hor):
        collection.append(Point(x_subset[j], t_subset[j], "", 0))

    # Split train and test set
    ts_train = collection[:-hor]
    ts_test = collection[length-1:]

    # Train model and predict test set
    model = EDM()
    model.train(ts_train, max_dim=int(np.sqrt(length)))
    _, smap = model.predict(ts_test, hor=hor)
    smap = smap.dropna(how='any')
    smap = smap[smap['hor'] == hor]

    # Measure performance
    try:
        RMSE = np.sqrt((smap['obs'] - smap['pred'])**2)[0]
        time = smap['time_stamp'][0]
    except:
        RMSE = np.sqrt((smap['obs'] - smap['pred']) ** 2).values[0]
        time = smap['time_stamp'].values[0]

    return RMSE, time

def create_plots_RMSE(x, y, z, t, RMSEs, ts, length, noise, hor):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot shadow of trajectory
    ax.plot(x, y, z, alpha=.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # define color mapping function
    max_rmse = np.percentile(RMSEs, 95)

    # add points to the plot
    dict = pd.DataFrame({'x':x, 'y':y, 'z':z, 't':t})
    for i in range(len(ts)):
        time_stamp = ts[i]
        rmse = RMSEs[i]
        x = dict[dict['t'] == time_stamp].iloc[0,0]
        y = dict[dict['t'] == time_stamp].iloc[0,1]
        z = dict[dict['t'] == time_stamp].iloc[0,2]
        color = plt.cm.YlOrRd(rmse/max_rmse)

        ax.scatter(x, y, z, c=color, marker='s', s=2)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=max_rmse))
    sm.set_array([])
    fig.colorbar(sm, label='RMSE')

    plt.tight_layout()
    plt.savefig(f'C:/Users/5605407/Documents/PhD/Chapter_1/Resultaten/RMSE on the attractor/len={length}, noise={noise}, hor={hor}.png', dpi=500)

def loop(noise, length, hor):
    rho = 28
    lorenz_length = 3000

    # Sample Lorenz trajectory
    x, y, z, t = simulate_lorenz([-10.59488751,-15.39807062,22.89394584], [10.0, rho, 8.0/3.0], lorenz_length * 5, lorenz_length * 5 / 1000, noise)
    x, y, z, t = sample_lorenz(x, y, z, t)
    t = np.arange(lorenz_length)

    # Define partial function
    RMSEs, ts = [], []
    for i in range(lorenz_length-length-hor):
        RMSE, time_stamp = get_one_RMSE(i, x, t, length, hor)
        RMSEs.append(RMSE)
        ts.append(time_stamp)

    # Plot in 3D
    create_plots_RMSE(x, y, z, t, RMSEs, ts, length, noise, hor)


if __name__ == '__main__':

    noise = 0.0
    for hor in [3, 4, 5, 6, 7, 8, 9, 10]:
        for length in [25, 50, 75, 100, 150]:
            loop(noise, length, hor)

