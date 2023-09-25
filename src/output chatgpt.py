import numpy as np

def GPEDM_predict(pars, pop, xd, yd, Fig, predgrid, popgrid, EmbedDim, stepsahead, rhofixed):
    # Covariance function is exponential (dexp=1) or squared-exponential (dexp=2)
    # pop is vector identifying separate populations
    d = EmbedDim
    h = stepsahead
    T = len(yd)
    Y = yd
    rho = None
    dexp = 2
    if len(np.unique(pop)) == 1:
        pars[d + 3] = 0

    if Fig == 4:
        td = xd[:, 0]
        xd = xd[:, 1:]  # a work-around to include time data for sequential updates

    # transform parameters from real line to constrained space
    vemin = 0.01
    taumin = 0.01
    vemax = 4.99
    taumax = 4.99
    rhomin = 0.0001
    rhomax = 1
    phi = np.exp(pars[:d])  # phi = 0.1 * np.ones(d)
    ve = (vemax - vemin) / (1 + np.exp(-pars[d + 1])) + vemin
    tau = (taumax - taumin) / (1 + np.exp(-pars[d + 2])) + taumin
    rho = (rhomax - rhomin) / (1 + np.exp(-pars[d + 3])) + rhomin

    if rhofixed is not None:
        rho = rhofixed

    # [phi, tau, ve, rho]

    # derivative for rescaled parameters wrt inputs -- for gradient calculation
    dpars = np.concatenate([phi, (ve - vemin) * (1 - (ve - vemin) / (vemax - vemin)),
                            (tau - taumin) * (1 - (tau - taumin) / (taumax - taumin)),
                            (rho - rhomin) * (1 - (rho - rhomin) / (rhomax - rhomin))])

    # specify priors
    # length scale
    lam_phi = (2 ** (stepsahead - 1)) ** 2 * np.pi / 2  # variance for gaussian - pi/2 means E(phi)=1
    lp_phi = -0.5 * np.sum(phi ** 2) / lam_phi
    dlp_phi = -phi / lam_phi

    # Rest of your code...
