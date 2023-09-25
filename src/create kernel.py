import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import special
from scipy import stats

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Alpha: pass an array of length equal to the number of entreis. Can be interpreted as the variance of additional
# Gaussian measurement noise on the training observations.

# set normalize_y to True if zero-mean, unit-variance priors are used.

# log_marginal_likelihood_value ?

# When training a Gaussian process, the hyperparameters of the kernel are optimized during the fitting process.

def get_length_scale_parameters(ts, phi):
    r = max(ts) - min(ts)
    l = np.sqrt(2) * phi/r
    return l

def set_kernel(tau, alpha, noise_term=1e-1):
    return tau**2 * RBF(length_scale=alpha)

def gpr(kernel, X_train, y_train, X_test, y_test):
    gpr = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
    predictions, std = gpr.predict(X_test, return_std=True)
    gpr.kernel_


def inverse_cdf(y):
    # Computed analytically
    return math.sqrt(math.log(-1/(y - 1)))

def sample_distribution():
    uniform_random_sample = random.random()
    return inverse_cdf(uniform_random_sample)


x = [sample_distribution() for i in range(10000)]
plt.hist(x, bins=50)
plt.show()


if __name__ == "__main__":
    E = 3
    my_rv = RandomPhi()
    samples = my_rv.rvs(size=E)

    # plot histogram of samples
    fig, ax1 = plt.subplots()
    ax1.hist(list(samples), bins=50)

    # plot PDF and CDF of distribution
    pts = np.linspace(0, 5)
    ax2 = ax1.twinx()
    ax2.set_ylim(0, 1.1)
    ax2.plot(pts, my_rv.pdf(pts), color='red')
    ax2.plot(pts, my_rv.cdf(pts), color='orange')

    fig.tight_layout()
    plt.show()
