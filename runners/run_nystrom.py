import numpy as np
from scipy import optimize
from scipy.linalg import solve
from scipy.stats import norm, uniform
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, cdist, squareform
from modules.nystrom import xx_nystrom, xu_nystrom, nystorm_2d
from modules.kernel import RBFKernel

K = RBFKernel()


def loss(alpha, obs_size, mat_k_xx, a, gamma):
    loss_value = 0
    for i in range(obs_size):
        loss_value -= np.log(a * (alpha @ mat_k_xx[i]) ** 2)
    loss_value += gamma * alpha @ mat_k_xx @ mat_k_xx.T @ alpha
    return loss_value


def loss_gradient(alpha, obs_size, mat_k_xx, a, gamma):
    gradient_value = np.zeros(obs_size)
    for i in range(obs_size):
        gradient_value -= 2 * mat_k_xx[i] / (alpha @ mat_k_xx)
    gradient_value += 2 * gamma * alpha @ mat_k_xx
    return gradient_value


def run_nystrom(grid, obs, a, gamma):
    mat_k_xx = xx_nystrom(grid, obs, a, gamma)
    grid_size = len(grid)
    obs_size = len(obs)
    alpha_init = norm.rvs(0, 1, obs_size)
    alpha_new = optimize.minimize(loss, alpha_init, args=(obs_size, mat_k_xx, a, gamma), method='SLSQP',
                                  jac=loss_gradient, bounds=[(-grid_size, grid_size)] * obs_size, options={'disp': True})
    intensity = a * (alpha_new.x @ mat_k_xx) ** 2  # af^2(x), link function is quadratic function
    return intensity


def run_predict(grid, obs, region, a, gamma):
    mat_k_xx = xx_nystrom(region, obs, a, gamma)
    mat_k_xu = xu_nystrom(grid, obs, a, gamma)
    obs_size = len(obs)
    region_size = len(region)
    alpha_init = norm.rvs(0, 1, obs_size)
    alpha_new = optimize.minimize(loss, alpha_init, args=(obs_size, mat_k_xx, a, gamma), method='SLSQP',
                                  jac=loss_gradient, bounds=[(-region_size, region_size)] * obs_size, options={'disp': True})
    intensity = a * (alpha_new.x @ mat_k_xu) ** 2
    return intensity


def run_nystrom_2d(x_grid, y_grid, obs, a, gamma, num_samples=400):
    x, y = np.meshgrid(x_grid, y_grid)
    xy_grid = np.column_stack((x.ravel(), y.ravel()))[:, ::-1]
    sample_indices = np.random.choice(xy_grid.shape[0], num_samples, replace=False)
    xy_grid_sample = xy_grid[sample_indices, :]
    mat_k_xx, mat_k_xu = nystorm_2d(obs, xy_grid_sample, xy_grid, a, gamma, lowrank=10)
    obs_size = obs.shape[0]
    alpha_init = norm.rvs(0, 1, obs_size)
    alpha_new = optimize.minimize(loss, alpha_init, args=(obs_size, mat_k_xx, a, gamma), method='SLSQP',
                                  jac=loss_gradient, bounds=[(-x_grid.shape[0], y_grid.shape[0])] * obs_size, options={'disp': True})
    intensity = a * (alpha_new.x @ mat_k_xu) ** 2
    mat_intensity = np.reshape(intensity, (x_grid.shape[0], y_grid.shape[0])).T
    return intensity, mat_intensity
