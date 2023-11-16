import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import pathlib
import os

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import mapclassify as mc

from modules.dataloader import load_tornado_data
from runners.run_bayesian import run_bayesian
from time import perf_counter

save_dir = '../results/2022_tornadoes'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir_map = '../results/2022_tornadoes/heatmap'
if not os.path.exists(save_dir_map):
    os.makedirs(save_dir_map)

gamma = 0.5
grid_min = 0
grid_max = 365
grid_size = 730
grid = np.linspace(grid_min, grid_max, grid_size)

region_radius = 10
region_len = 2 * region_radius
grid_value_ratio = int(grid_size / grid_max)
region_size = region_len * grid_value_ratio

arr, arr_size, location = load_tornado_data()
print('Tornado counts: ', arr_size)

a = 0.02
step = 25
obs_hist = []
region_center = np.array([80.0, 160.0, 240.0])
region_mask = np.array(False).repeat(grid_size)

time = 0
time_step = []

for i in range(step):
    time_start = perf_counter()
    mean, mat_cov, std, region_center, obs_hist, acquisition_value, region_mask = run_bayesian(i, grid, grid_min, grid_max, region_mask, grid_value_ratio, region_center, region_radius, arr, obs_hist, a, gamma)
    time_end = perf_counter()
    print('Time for step %d: ' % (i + 1), time_end - time_start)
    time += (time_end - time_start)
    time_step.append(time_end - time_start)

    fig = plt.figure(1, figsize=(8, 4))
    axs = plt.subplot(111)
    axs.clear()
    print("Step: ", i + 1)
    print("Region centers: ", region_center[:-1])
    print("Max AF value: ", max(acquisition_value), "Next region center: ", region_center[-1])

    axs.plot(arr, np.zeros(arr_size), linestyle='None', marker='|', color='k', markersize=18)
    axs.plot(obs_hist, np.zeros(len(obs_hist)) - 0.2, linestyle='None', marker='|', color='r', markersize=18)
    axs.plot(grid, acquisition_value, label='UCB', color='g', linewidth=2)
    axs.plot(grid, mean, label='Ours', color='r', linewidth=2)
    plt.fill(np.concatenate([grid, grid[::-1]]),
             np.concatenate([np.where(mean - 1.9600 * std > 0, mean - 1.9600 * std, 0), (mean + 1.9600 * std)[::-1]]),
             alpha=.4, fc='r', ec='None', label='Variance')
    axs.tick_params(labelsize=18)
    plt.ylabel('Intensity: λ(t)', fontsize=18)
    plt.xlabel('Input: t', fontsize=18)
    if i == 0:
        plt.legend(loc=2, fontsize=18)
    # plt.show()
    # fig.savefig(save_dir + '/step_%d.png' % (i + 1), format='png', bbox_inches='tight', dpi=300)

    tornado_gdf = gpd.GeoDataFrame(location, geometry=gpd.points_from_xy(location['slon'], location['slat']))
    obs_indices = np.in1d(arr.reshape(-1), np.array(obs_hist))
    obs_intensity = mean[np.array(obs_hist).astype('int') * grid_value_ratio]
    tornado_gdf['intensity'][obs_indices] = obs_intensity  # replace default intensities with obs intensities

    contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
    axs_usa = gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea(), figsize=(4.5, 3), zorder=1, linewidth=0.7, alpha=0.5)
    gplt.kdeplot(tornado_gdf[tornado_gdf['intensity'] / max(tornado_gdf['intensity']) > 0.75], projection=gcrs.AlbersEqualArea(),
                 ax=axs_usa, cmap='Reds', fill=True, thresh=0, levels=100, legend=True, cut=6, alpha=1)
    gplt.polyplot(contiguous_usa, projection=gcrs.AlbersEqualArea(), ax=axs_usa, zorder=1, linewidth=0.7, alpha=0.5)
    gplt.pointplot(tornado_gdf[tornado_gdf['intensity'] / max(tornado_gdf['intensity']) > 0.25], projection=gcrs.AlbersEqualArea(),
                   ax=axs_usa, hue='intensity', cmap='Reds', s=0, legend=True)
    gplt.pointplot(tornado_gdf[obs_indices], ax=axs_usa, marker='.', color='k', s=12, edgecolor='lightgray', linewidth=1)
    axs_usa.set_xlim(-2.4e6, 2.3e6)
    axs_usa.set_ylim(-1.35e6, 1.7e6)
    plt.show()
    # axs_usa.figure.savefig(save_dir_map + '/step_%d.png' % (i + 1), format='png', bbox_inches='tight', dpi=300)
    # np.savetxt(save_dir + '/center.txt', region_center)
    # np.savetxt(save_dir + '/metric.txt', np.array([time_step]).T, delimiter=',')
    axs_usa.clear()

print('Total time: ', time)
np.save(save_dir + '/time.npy', np.array(time))
