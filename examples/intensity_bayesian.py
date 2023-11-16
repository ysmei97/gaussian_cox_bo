import numpy as np
import matplotlib.pyplot as plt
import os
from time import perf_counter

from modules.kernel import RBFKernel
from modules.bayesian import AcquisitionFunction, BayesianOptimizer, init_observation
from modules.generator import gt_intensity, thinning
from runners.run_bayesian import run_bayesian

K = RBFKernel()
AF = AcquisitionFunction()
BO = BayesianOptimizer(AF.upper_confidence_bound)

save_dir = '../results/1d_syn_bo'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gamma = 0.5
grid_min = 0
grid_max = 100
grid_size = 200
grid = np.linspace(grid_min, grid_max, grid_size)

region_radius = 2
region_len = 2 * region_radius
grid_value_ratio = int(grid_size / grid_max)
region_size = region_len * grid_value_ratio

# Ground truth intensity and arrivals
ground_truth = gt_intensity(grid, 64, 1.6)
# np.save(save_dir + '/ground_truth.npy', ground_truth)
intensity_max = max(ground_truth)
print('Intensity maximum: ', intensity_max)
arr, arr_size = thinning(ground_truth, intensity_max, grid_max)
print('Arrival counts: ', arr_size)

# Bayesian optimization
a = 0.04
step = 40
obs_hist = []
region_center = np.array([25.0, 60.0])
region_mask = np.array(False).repeat(grid_size)  # for computing variance, give 0 to positions outside obs regions

time = 0

for i in range(step):
    time_start = perf_counter()
    intensity_prediction, mat_cov, std, region_center, obs_hist, acquisition_value, region_mask = run_bayesian(i, grid, grid_min, grid_max, region_mask, grid_value_ratio, region_center, region_radius, arr, obs_hist, a, gamma)
    time_end = perf_counter()
    print('Time for step %d: ' % (i + 1), time_end - time_start)
    time += (time_end - time_start)

    fig = plt.figure(1, figsize=(8, 4))
    axs = plt.subplot(111)
    axs.clear()

    print(region_center)
    axs.plot(grid, ground_truth, label='Truth', color='k', linewidth=2)
    axs.plot(arr, np.zeros(arr_size), linestyle='None', marker='|', color='k', markersize=18)
    axs.plot(grid, intensity_prediction, label='Ours', color='r', linewidth=2)
    axs.plot(obs_hist, np.zeros(len(obs_hist)) - 0.8, linestyle='None', marker='|', color='r', markersize=18)
    plt.fill(np.concatenate([grid, grid[::-1]]),
             np.concatenate([np.where(intensity_prediction - 1.9600 * std > 0, intensity_prediction - 1.9600 * std, 0),
                             (intensity_prediction + 1.9600 * std)[::-1]]),
             alpha=.4, fc='r', ec='None', label='Variance')
    axs.plot(grid, acquisition_value, label='UCB', color='g', linewidth=2)
    axs.tick_params(labelsize=18)
    plt.ylabel('Intensity: λ(t)', fontsize=18)
    plt.xlabel('Input: t', fontsize=18)
    if i == 0:
        plt.legend(loc=2, fontsize=18)
    plt.show()

    # fig.savefig(save_dir + '/step_%d.png' % (i + 1), format='png', bbox_inches='tight', dpi=300)
    # np.savetxt(save_dir + '/center.txt', region_center)

print("Time: ", time)