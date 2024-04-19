import matplotlib.pyplot as plt

from src.simulate_lorenz import *
from src.classes import *

def sample_lorenz(x, y, z, t):
    x_ = [x[i] for i in range(len(x)) if i % 5 == 0]
    y_ = [y[i] for i in range(len(x)) if i % 5 == 0]
    z_ = [z[i] for i in range(len(x)) if i % 5 == 0]
    t_ = [t[i] for i in range(len(t)) if i % 5 == 0]

    return x_, y_, z_, t_

def create_plots(rho):
    # get initial point
    x, y, z, t = simulate_lorenz([1, 1, 1], [10.0, rho, 8.0 / 3.0],1000 * 5, 1000 * 5 / 1000, 0.0)
    vec0 = [x[-1], y[-1], z[-1]]

    x, y, z, t = simulate_lorenz(vec0, [10.0, rho, 8.0 / 3.0], 3000 * 5, 3000 * 5 / 1000, 0.0)
    x, y, z, t = sample_lorenz(x, y, z, t)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot shadow of trajectory
    ax.plot(x, y, z, alpha=.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # add points to the plot
    for i in range(50):
        index = np.random.randint(0, len(x)-1)
        x_ = x[index]
        y_ = y[index]
        z_ = z[index]
        ax.scatter(x_, y_, z_, s=30)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    plt.tight_layout()

    plt.savefig('rho = 28.png', dpi=500)

    plt.show()

    # plot x, y and z time series
    plt.rc('font', size=15)

    x, t = sample_from_ts(x, t, 5)
    plt.plot(t[-50:], x[-50:], zorder=0, color='grey')

    x[-50:] += np.random.normal(0, 1.5, size=len(x[-50:]))
    plt.scatter(t[-50:], x[-50:])
    plt.plot(t[-50:], x[-50:])

    plt.xlabel('time')
    plt.ylabel('x')

    plt.xticks([])
    plt.yticks([])

    plt.show()


if __name__ == "__main__":
    create_plots(28)