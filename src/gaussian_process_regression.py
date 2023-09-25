from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
from create_dummy_time_series import *
from empirical_dynamic_modeling import embed_time_series

# from https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_prior_posterior.html#sphx-glr-auto-examples-gaussian-process-plot-gpr-prior-posterior-py


def plot_univariate_gpr_samples(prior, post, n_samples, min=-20, max=20):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    """
    # Create grid
    x = np.linspace(min, max, 100)
    X = x.reshape(-1, 1)


    y_post_mean, y_post_std = post.predict(X, return_std=True)
    y_post_samples = post.sample_y(X, n_samples)


    y_prior_mean, y_prior_std = prior.predict(X, return_std=True)
    y_prior_samples = prior.sample_y(X, n_samples)


    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))

    for idx, single_prior in enumerate(y_prior_samples.T):
        axs[0].plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )

    axs[0].plot(x, y_prior_mean, color="black", label="Mean")
    axs[0].fill_between(
        x,
        y_prior_mean - y_prior_std,
        y_prior_mean + y_prior_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    axs[0].set_xlabel("x(t-1)")
    axs[0].set_ylabel("x(t) (predicted)")
    axs[0].set_ylim([-4, 4])

    for idx, single_post in enumerate(y_post_samples.T):
        axs[1].plot(
            x,
            single_post,
            linestyle="--",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
        )

    axs[1].plot(x, y_post_mean, color="black", label="Mean")
    axs[1].fill_between(
        x,
        y_post_mean - y_post_std,
        y_post_mean + y_post_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 std. dev.",
    )
    axs[1].set_xlabel("x(t-1)")
    axs[1].set_ylabel("x(t) (predicted)")
    axs[1].set_ylim([-21, 20])

    fig.suptitle("Gaussian Process Regression - prior and posterior")
    fig.show()


def plot_2d_gpr_samples(prior, post):
    return 0


# def difference(ts, interval=1):
#     diff = []
#     for i in range(interval, len(ts)):
#         value = ts[i] - ts[i - interval]
#         diff.append(value)
#     return diff


if __name__ == "__main__":

    # Set parameters
    E = 3


    # Simulate the Lorenz System
    x_, y_, z_, t_ = simulate_lorenz(tmax=40, ntimesteps=1500, obs_noise_sd=0)

    # Change sampling interval
    spin_off = 300
    sampling_interval = 5

    x = [x_[i] for i in range(1, len(x_)) if i%sampling_interval==0 and i > spin_off]
    t = [t_[i] for i in range(1, len(x_)) if i%sampling_interval==0 and i > spin_off]


    # Embed time series
    lib = embed_time_series(x, lag=1, E=E)


    # Split predictor from response variables
    X, y = [], []

    for point in lib:
        X.append(point[0])
        y.append(point[1])


    # Split into training and test set (time ordered)
    cut_off = int(len(X) * 0.7)
    X_train, y_train = X[:cut_off], y[:cut_off]
    X_test, y_test = X[cut_off+1:], y[cut_off+1:]


    # Plot time series
    plt.plot(t_, x_, color='grey', linewidth=1, linestyle='--')
    plt.plot(t, x, color='orange', linewidth=2)
    plt.scatter(t, x, color='red')
    plt.axvline(x=t[0], color='red')
    plt.axvline(x=((t[cut_off] + t[cut_off + 1]) / 2.0), color='red')
    plt.title('Input time series')
    plt.show()


    print("Number of training samples: " + str(len(X_train)))
    print("Number of test samples: " + str(len(X_test)))
    print("Sampling interval: " + str(t[1] - t[0]))


    # Not time ordered:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


    # Transform into array-likes
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)


    # Create kernel
    length_scales = np.ones(E)
    kernel = 1.0 * RBF(length_scales)

    # alpha
    # the variance of additional Gaussian measurement noise on the training observations

    gpr_prior = GaussianProcessRegressor(kernel=kernel,
                                         normalize_y = True,
                                         random_state=0)


    gpr = GaussianProcessRegressor(kernel=kernel,
                                   normalize_y=True,
                                   random_state=0).fit(X_train, y_train)


    score = gpr.score(X_train, y_train)
    print("Optimized length scales: " + str(gpr.kernel_))

    pred_test = gpr.predict(X_test, return_std=True)

    plt.plot(np.linspace(min(y_test) - 1, max(y_test) + 1, num=50), np.linspace(min(y_test) - 1, max(y_test) + 1, num=50))
    plt.scatter(y_test, pred_test[0])
    plt.title("GPR one-step ahead prediction")
    plt.xlabel("observed")
    plt.ylabel("predicted")
    plt.show()

    if E == 1:
        plot_univariate_gpr_samples(gpr_prior, gpr, 3)

    # fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(10, 8))
    # plot_gpr_samples(min(X_train)[0], max(X_train)[0], gpr, n_samples=1, ax=axs[1])
    # axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
    # axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
    # axs[1].set_title("Samples from posterior distribution")
    #
    # fig.suptitle("Radial Basis Function kernel", fontsize=18)
    # plt.tight_layout()
    # fig.show()





