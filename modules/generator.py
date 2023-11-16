import numpy as np
from scipy.stats import norm, poisson, uniform, expon
from modules.kernel import RBFKernel

K = RBFKernel()


def gt_intensity(grid, k, a):
    """
    Generate ground-truth intensity function
    :param grid:
    :param k: Hermite polynomial order
    :param a:
    :return:
    """
    gt_alpha = norm.rvs(0, 1, k)
    k_sum = 0
    for i in range(k):
        eta_i = K.rbf_eigenvalue(i)
        psi_i = K.rbf_eigenfunction(i, grid)
        k_sum += gt_alpha[i] * np.sqrt(eta_i) * psi_i
    intensity = a * k_sum ** 2
    return intensity


def poisson_obs(intensity_max, time_max, time_min=0):
    obs_size = poisson.rvs(intensity_max)  # Random Poisson experiment, default size=1
    obs_time = uniform.rvs(time_min, time_max, obs_size)
    obs_time.sort()
    return obs_time, obs_size


def periodic_obs(obs_size, time_max, time_min=0):
    obs_time = np.linspace(time_min, time_max, obs_size)
    return obs_time


def random_obs(obs_size, time_max, time_min=0):
    obs_time = uniform.rvs(time_min, time_max, obs_size)
    obs_time.sort()
    return obs_time


def thinning(intensity, intensity_max, time_max, time_min=0):
    arr = time_min
    arr_time = []
    while arr < time_max:
        arr += expon.rvs(scale=1/intensity_max)
        if arr >= time_max:
            break
        rand_prob = uniform.rvs()
        if rand_prob * intensity_max <= intensity[int(arr)]:
            arr_time.append(arr)
    return np.array(arr_time), len(arr_time)
