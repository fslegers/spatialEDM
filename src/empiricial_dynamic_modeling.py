import skedm.data as data
import matplotlib.pyplot as plt

def uniform_embed(ts, tau, m):
    """
    :param ts: a time series (with uniform time intervals)
    :param tau: uniform lag of the embedding
    :param m: dimension of the embedding
    :return: an embedding of the time series by creating dim time series, each one shifted backwards lag time steps.
    """

    # discard the first (oldest) tau(m-1) observations
    n = len(ts)

    # add to embedding matrix
    TS = []

    # shift time series and add to TS
    for i in range(m):
        ts_lagged = ts[((m - (i + 1))*tau):(n - i*tau)]
        TS.append(ts_lagged)

    return TS

if __name__ == "__main__":

    # get x, y and z values of lorenz system with step size = 0.1 and tmax = 10.000
    X = data.lorenz(sz = 10000)[:,0]
    Y = data.lorenz(sz = 10000)[:,1]
    Z = data.lorenz(sz = 10000)[:,2]

    X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    uniform_embed(X, 3, 3)














