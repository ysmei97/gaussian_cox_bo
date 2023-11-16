import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os
from runners.run_bayesian import run_bayesian
from runners.run_nystrom import run_nystrom_2d


save_dir = '../results/real_world/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

gamma = 0.5
grid_min = 0
grid_max = 1
grid_size = 100
x_grid = np.linspace(grid_min, grid_max, grid_size)
y_grid = np.linspace(grid_min, grid_max, grid_size)

region_radius = 20
region_len = 2 * region_radius
grid_value_ratio = int(grid_size / grid_max)
region_size = region_len * grid_value_ratio

arr = np.load('../dataset/real_world/neuron.npy')[:]
lon = arr[:, 0]
lat = arr[:, 1]
lon = (lon - min(lon)) / (max(lon) - min(lon))
lat = (lat - min(lat)) / (max(lat) - min(lat))
location = np.concatenate((lon[:, None], lat[:, None]), axis=1)


obs_hist = []
region_center = np.array([20.0])
region_mask = np.array(False).repeat(grid_size)

intensity_list = []
for _ in range(1):
    intensity, mat_intensity = run_nystrom_2d(x_grid, y_grid, location, 8, gamma, 4000) # kernel (1, 50, 0.01)
    intensity_list.append(intensity)
intensity_avg = np.average(intensity_list, axis=0)
intensity_avg = np.array([0.0001 if i < 0 else i for i in intensity_avg])
intensity_max = max(intensity_avg)
intensity_avg = np.reshape(intensity_avg, (x_grid.shape[0], y_grid.shape[0])).T

fig = plt.figure(figsize=(2, 2))
axs = fig.add_axes((0.1, 0.1, 0.8, 0.8))
x, y = np.meshgrid(x_grid, y_grid)
axs.pcolor(x, y, intensity_avg, vmax=intensity_max, rasterized=1, cmap='Reds')
axs.scatter(location[:, 0], location[:, 1], s=2, c='k')
axs.set_xticks([])
axs.set_yticks([])
axs.set(xlabel=None, ylabel=None)
axs.set_xlim(x_grid[0], x_grid[-1])
axs.set_ylim(y_grid[0], y_grid[-1])
plt.show()
# fig.savefig(save_dir + '/neuron.png', format='png', bbox_inches='tight', dpi=300)

