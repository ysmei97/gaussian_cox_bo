import numpy as np
import matplotlib.pyplot as plt
import os

from modules.kernel import RBFKernel
from modules.generator import gt_intensity, thinning
from modules.bayesian import AcquisitionFunction, BayesianOptimizer
from modules.dataloader import load_coal_mine_disaster_data
from modules.bayesian import grf_predict
from runners.run_bayesian import run_bayesian
from time import perf_counter

save_dir = '../results/coal_mine_disasters'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gamma = 0.5
grid_min = 0
grid_max = 112
grid_size = 200
grid = np.linspace(grid_min, grid_max, grid_size)
a = 0.02
region_radius = 4
region_len = 2 * region_radius
grid_value_ratio = int(grid_size / grid_max)
region_size = region_len * grid_value_ratio

# Mining disasters
arr, arr_size = load_coal_mine_disaster_data()
print('Disaster counts: ', arr_size)

step = 10
obs_hist = []
region_center = np.array([20.0, 60.0])
region_mask = np.array(False).repeat(grid_size)

time = 0

for i in range(step):
    time_start = perf_counter()
    mean, mat_cov, std, region_center, obs_hist, acquisition_value, region_mask = run_bayesian(i, grid, grid_min, grid_max, region_mask, grid_value_ratio, region_center, region_radius, arr, obs_hist, a, gamma)
    time_end = perf_counter()
    print('Time for step %d: ' % (i + 1), time_end - time_start)
    time += (time_end - time_start)

    fig = plt.figure(1, figsize=(8, 3))
    axs = plt.subplot(111)
    axs.clear()
    print("Step: ", i + 1)
    print("Region centers: ", region_center[:-1])
    print("Max AF value: ", max(acquisition_value), "Next region center: ", region_center[-1])

    axs.plot(arr, np.zeros(arr_size), linestyle='None', marker='|', color='k', markersize=18)
    axs.plot(obs_hist, np.zeros(len(obs_hist)) - 0.2, linestyle='None', marker='|', color='r', markersize=18)
    # axs.plot(grid, acquisition_value, label='Idle', color='g', linewidth=2)
    axs.plot(grid, acquisition_value * 0.5, label='Cum', color='g', linewidth=2)
    # axs.plot(grid, acquisition_value, label='CPD', color='g', linewidth=2)
    axs.plot(grid, mean, label='Ours', color='r', linewidth=2)
    plt.fill(np.concatenate([grid, grid[::-1]]),
             np.concatenate([np.where(mean - 1.9600 * std > 0, mean - 1.9600 * std, 0), (mean + 1.9600 * std)[::-1]]),
             alpha=.4, fc='r', ec='None', label='Variance')
    axs.tick_params(labelsize=18)
    plt.ylabel('Intensity: λ(t)', fontsize=18)
    plt.xlabel('Input: t', fontsize=18)
    if i == 0:
        plt.legend(loc=2, fontsize=18)
    plt.show()

    # fig.savefig(save_dir + '/step_%d.png' % (i + 1), format='png', bbox_inches='tight', dpi=300)
    # np.savetxt(save_dir + '/center.txt', region_center)

print("Time: ", time)
# np.save(save_dir + '/time.npy', np.array(time))
