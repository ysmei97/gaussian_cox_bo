import numpy as np
from scipy.linalg import eigh
from scipy.linalg import solve
from scipy.stats import norm, uniform
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, cdist, squareform
from modules.kernel import RBFKernel

K = RBFKernel()


def xx_nystrom(grid, obs, a, gamma):
    grid_size = len(grid)
    grid = grid.reshape(1, -1)
    obs = obs.reshape(1, -1)
    mat_k_uu = K.rbf_vector(grid, grid)
    mat_k_xu = K.rbf_vector(obs, grid)
    eigenvalue, eigenvector = eigh(mat_k_uu)
    eigenvalue[eigenvalue <= 0] = 1.e-4
    mat_eigenvalue = np.diag(eigenvalue)
    mat_eigenvalue_est_inv = np.linalg.inv(a * mat_eigenvalue ** 2 / grid_size + gamma * mat_eigenvalue)
    mat_k_xx = mat_k_xu @ eigenvector @ mat_eigenvalue_est_inv @ eigenvector.T @ mat_k_xu.T
    return mat_k_xx


def xu_nystrom(grid, obs, a, gamma):
    grid_size = len(grid)
    grid = grid.reshape(1, -1)
    obs = obs.reshape(1, -1)
    mat_k_uu = K.rbf_vector(grid, grid)
    mat_k_xu = K.rbf_vector(obs, grid)
    mat_k_uu[mat_k_uu <= 0] = 1.e-4
    return mat_k_xu @ np.linalg.inv(a * mat_k_uu / grid_size + gamma * np.diag(np.ones(grid_size)))


def nystorm_2d(x, u, grid, a, gamma, lowrank=10):
    mat_k_uu = K.rbf_dist(cdist(u, u))
    mat_k_grid = K.rbf_dist(cdist(grid, grid))
    mat_k_xu = K.rbf_dist(cdist(x, u))
    mat_k_xgrid = K.rbf_dist(cdist(x, grid))
    eigenvalue, eigenvector = eigsh(mat_k_uu, k=lowrank)
    mat_eigenvalue = np.diag(eigenvalue)
    mat_eigenvalue_est_inv = np.linalg.inv(a * mat_eigenvalue ** 2 / u.shape[0] + gamma * mat_eigenvalue)
    mat_k_xx = mat_k_xu @ eigenvector @ mat_eigenvalue_est_inv @ eigenvector.T @ mat_k_xu.T
    mat_k_xgrid_ = solve(a * mat_k_grid / grid.shape[0] + gamma * np.eye(grid.shape[0]), mat_k_xgrid.T).T
    return mat_k_xx, mat_k_xgrid_
