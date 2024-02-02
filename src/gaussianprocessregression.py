import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from pymc.gp.util import plot_gp_dist

np.random.seed(42)

# Generate synthetic data
X = np.sort(5 * np.random.rand(20))[:, np.newaxis]
y = np.sin(X).ravel() + 0.1 * np.random.randn(20)

with pm.Model() as gp_model:
    # Define hyperparameters for the kernel
    length_scale = pm.Gamma("length_scale", alpha=2, beta=1)
    noise = pm.HalfCauchy("noise", beta=5)

    # Define the RBF kernel
    cov_func = pm.gp.cov.ExpQuad(1, ls=length_scale)

    # Create Gaussian Process
    gp = pm.gp.Marginal(cov_func=cov_func)

    # Define the GP prior
    y_ = gp.marginal_likelihood("y", X=X, y=y, noise=noise)

with gp_model:
    trace = pm.sample(1000, tune=1000, cores=1)

pm.plot_posterior(trace, var_names=["length_scale", "noise"])
plt.show()

# Make predictions with the trained model
X_new = np.linspace(0, 5, 100)[:, np.newaxis]

with gp_model:
    y_pred = gp.conditional("y_pred", X_new)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Observed data")
pm.plot_posterior_predictive(trace, var_names=["y_pred"], samples=100, color="red", alpha=0.5)
plt.plot(X_new, np.mean(trace["y_pred"], axis=0), color="red", label="Mean prediction")
plt.title("Gaussian Process Regression with RBF Kernel")
plt.legend()
plt.show()