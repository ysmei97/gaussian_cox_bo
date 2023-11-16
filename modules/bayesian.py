import numpy as np
from functools import partial
from scipy.stats import norm
from scipy.special import erf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, WhiteKernel, Matern

from modules.kernel import RBFKernel
from modules.generator import thinning
from runners.run_nystrom import run_nystrom, run_predict
from utils.change_point import online_changepoint_detection
from utils.online_likelihoods import StudentT

K = RBFKernel()


def grf_predict(grid, obs, intensity, region_mask, grid_value_ratio, link_func_inv, link_func, link_func_grad_1, link_func_grad_2):
    """
    :param grid: column vector grid
    :param obs: column vector observation
    :param intensity:
    :param grid_value_ratio:
    :param link_func_inv: kappa inverse
    :param link_func: kappa
    :param link_func_grad_1: 1st order gradient of kappa
    :param link_func_grad_2: 2nd order gradient of kappa
    :return:
    """
    assert grid.shape[1] == 1
    assert obs.shape[1] == 1
    intensity_inv = link_func_inv(intensity)
    gp_model = GaussianProcessRegressor(kernel=RBF(), random_state=0)
    gp_model.fit(obs, intensity_inv)
    mean, cov = gp_model.predict(grid, return_cov=True)
    cov_inv = np.linalg.inv(cov)
    grid_interval = 1 / grid_value_ratio
    vec_cov_grad = (link_func(mean) * link_func_grad_2(mean) - link_func_grad_1(mean) ** 2) / link_func(mean) ** 2 - link_func_grad_2(mean) * grid_interval
    vec_cov_grad *= region_mask
    mat_cov_grad = np.diag(vec_cov_grad)
    mat_cov = np.linalg.inv(cov_inv - mat_cov_grad)
    std = np.sqrt(np.abs(np.diag(mat_cov))) * 1.0
    return mean, mat_cov, std


class AcquisitionFunction:
    def __init__(self, kappa=4.0, delta=0.0, beta=0.8):
        self.kappa = kappa
        self.delta = delta
        self.beta = beta

    def probability_of_improvement(self, mean, std, grid_max, region_rad, grid_value_ratio):
        """
        Returns the probability of improvement at query point
        :param mean:
        :param std:
        :return:
        """
        max_intensity = max(mean)
        z = (mean - (max_intensity + self.delta)) / std
        pi = norm.cdf(z)
        return pi

    def expected_improvement(self, mean, std, grid_max, region_rad, grid_value_ratio):
        """
        Returns the expected improvement at query point
        :param mean:
        :param std:
        :return:
        """
        max_intensity = max(mean)
        z = (mean - (max_intensity + self.delta)) / std
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        return ei

    def upper_confidence_bound(self, mean, std, grid_max, region_rad, grid_value_ratio):
        """
        Returns the upper confidence point of performance at query point
        :param mean:
        :param std:
        :return:
        """
        ucb = mean + self.kappa * std
        return ucb

    def idle_time_detection(self, mean, std, grid_max, region_rad, grid_value_ratio):
        """
        Detect the longest idle region in the whole time domain
        :param mean:
        :param std:
        :return:
        """
        max_intensity = max(mean + self.beta * std)
        arr_prediction, arr_prediction_size = thinning(mean + self.beta * std, max_intensity, grid_max)
        grid_size = len(mean)
        region_len = 2 * region_rad
        itd = np.zeros(grid_size)
        for i in range(grid_size):
            j = i + 2 * region_rad * grid_value_ratio
            if j == grid_size:
                break
            arr_i = arr_prediction[arr_prediction >= (i / grid_value_ratio)]
            arr_i = arr_i[arr_i <= (i / grid_value_ratio + region_len)]
            itd[i + region_rad * grid_value_ratio] = 1 / (sum(arr_i) + 1)
        return itd

    def cum_arr_detection(self, mean, std, grid_max, region_rad, grid_value_ratio):
        """
        Detect the region with the most cumulative arrivals
        :param mean:
        :param std:
        :return:
        """
        max_intensity = max(mean + self.beta * std)
        arr_prediction, arr_prediction_size = thinning(mean + self.beta * std, max_intensity, grid_max)
        grid_size = len(mean)
        region_len = 2 * region_rad
        cad = np.zeros(grid_size)
        for i in range(grid_size):
            j = i + 2 * region_rad * grid_value_ratio
            if j == grid_size:
                break
            arr_i = arr_prediction[arr_prediction >= (i / grid_value_ratio)]
            arr_i = arr_i[arr_i <= (i / grid_value_ratio + region_len)]
            cad[i + region_rad * grid_value_ratio] = sum(arr_i) / arr_prediction_size
        return cad / max(cad)

    def change_point_detection(self, mean, std, grid_max, region_rad, grid_value_ratio):
        """
        Detect the region where intensity changes drastically
        :param mean:
        :param std:
        :param grid_max:
        :param region_rad:
        :param grid_value_ratio:
        :return:
        """
        hazard = lambda lam, r: 1 / lam * np.ones(r.shape)
        hazard_func = partial(hazard, 250)
        cpd, maxes = online_changepoint_detection(mean + self.beta * std, hazard_func, StudentT(alpha=0.1, beta=.01, kappa=1, mu=0))
        return cpd[8, :-1]


class BayesianOptimizer:
    def __init__(self, acquisition_function):
        self.acquisition_function = acquisition_function
        self.kappa = lambda g: g ** 2
        self.kappa_inv = lambda g: np.sqrt(g)
        self.kappa_grad_1 = lambda g: 2 * g
        self.kappa_grad_2 = lambda g: 2

    def select_next_region(self, grid_min, grid_max, grid_value_ratio, region_center, region_rad, mean, std, intensity):
        acquisition_value = self.acquisition_function(mean, std, grid_max, region_rad, grid_value_ratio)
        for i in range(int(4 * region_rad * grid_value_ratio)):
            region_i = (region_center - 2 * region_rad) * grid_value_ratio + i
            region_i = region_i.astype(int)
            region_i = np.delete(region_i, region_i < grid_min * grid_value_ratio)
            region_i = np.delete(region_i, region_i >= grid_max * grid_value_ratio)
            acquisition_value[region_i] = 0.0
        next_region_center = np.argmax(acquisition_value) / grid_value_ratio
        region_center = np.append(region_center, next_region_center)
        return region_center, acquisition_value

    def update_model(self, grid, obs_hist, region_mask, grid_value_ratio, a, gamma):
        intensity_list = []
        for _ in range(10):
            intensity_list.append(run_nystrom(grid, np.array(obs_hist), a, gamma))
        intensity_avg = np.average(intensity_list, axis=0)
        intensity_avg = np.array([0.0001 if i < 0 else i for i in intensity_avg])
        intensity_list = []
        region = region_mask * grid
        for _ in range(10):
            intensity_list.append(run_predict(grid, np.array(obs_hist), region, a, gamma))
        intensity_mean = np.average(intensity_list, axis=0)
        intensity_mean = np.array([0.0001 if i < 0 else i for i in intensity_mean])
        grid = grid[:, None]
        obs = np.array(obs_hist)[:, None]
        intensity_avg = intensity_avg[:, None]
        intensity_prediction, mat_cov, std = grf_predict(grid, obs, intensity_avg, region_mask, grid_value_ratio, self.kappa_inv, self.kappa, self.kappa_grad_1, self.kappa_grad_2)
        std *= 1.0
        return intensity_mean, mat_cov, std, intensity_avg


def observe_arr_in_region(grid_min, grid_max, region_center, region_mask, region_rad, arr, obs_hist):
    new_region_center = region_center[-1]
    new_region_min = new_region_center - region_rad if new_region_center - region_rad > grid_min else grid_min
    new_region_max = new_region_center + region_rad if new_region_center + region_rad < grid_max else grid_max
    new_obs = arr[arr >= new_region_min]
    new_obs = new_obs[new_obs <= new_region_max]
    obs_hist += list(new_obs)
    region_mask[int(new_region_min):int(new_region_max + 1)] = True
    return obs_hist, region_mask


def init_observation(grid_min, grid_max, region_center, region_mask, region_rad, arr, obs_hist):
    init_region_num = len(region_center)
    for i in range(init_region_num):
        init_region_min = region_center[i] - region_rad if region_center[i] - region_rad > grid_min else grid_min
        init_region_max = region_center[i] + region_rad if region_center[i] + region_rad < grid_max else grid_max
        new_obs = arr[arr >= init_region_min]
        new_obs = new_obs[new_obs <= init_region_max]
        obs_hist += list(new_obs)
        region_mask[int(init_region_min):int(init_region_max + 1)] = True
    return obs_hist, region_mask
