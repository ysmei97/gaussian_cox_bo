import numpy as np

from modules.kernel import RBFKernel
from modules.bayesian import AcquisitionFunction, BayesianOptimizer, observe_arr_in_region, init_observation

K = RBFKernel()
AF = AcquisitionFunction(kappa=4.0, beta=0.8)
BO = BayesianOptimizer(AF.upper_confidence_bound)


def run_bayesian(step, grid, grid_min, grid_max, region_mask, grid_value_ratio, region_center, region_rad, arr, obs_hist, a, gamma):
    if step == 0:
        obs_hist, region_mask = init_observation(grid_min, grid_max, region_center, region_mask, region_rad, arr, obs_hist)
    else:
        obs_hist, region_mask = observe_arr_in_region(grid_min, grid_max, region_center, region_mask, region_rad, arr, obs_hist)
    intensity_pre, mat_cov, std, intensity = BO.update_model(grid, obs_hist, region_mask, grid_value_ratio, a, gamma)
    region_center, acquisition_value = BO.select_next_region(grid_min, grid_max, grid_value_ratio, region_center, region_rad, np.abs(intensity_pre), std, intensity)
    return intensity_pre, mat_cov, std, region_center, obs_hist, acquisition_value, region_mask
