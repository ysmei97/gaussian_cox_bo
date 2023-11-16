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

from modules.dataloader import load_crime_data
from runners.run_bayesian import run_bayesian
from time import perf_counter

save_dir = '../results/2022_dc_crime'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_dir_map = '../results/2022_dc_crime/heatmap'
if not os.path.exists(save_dir_map):
    os.makedirs(save_dir_map)

gamma = 0.5
grid_min = 0
grid_max = 61
grid_size = 122
grid = np.linspace(grid_min, grid_max, grid_size)

region_radius = 1
region_len = 2 * region_radius
grid_value_ratio = int(grid_size / grid_max)
region_size = region_len * grid_value_ratio

arr, arr_size, location = load_crime_data()
print('Crime counts: ', arr_size)

a = 0.02
step = 25
obs_hist = []
region_center = np.array([20.0, 40.0])
region_mask = np.array(False).repeat(grid_size)

time = 0

for i in range(step):
    time_start = perf_counter()
    mean, mat_cov, std, region_center, obs_hist, acquisition_value, region_mask = run_bayesian(i, grid, grid_min, grid_max, region_mask, grid_value_ratio, region_center, region_radius, arr, obs_hist, a, gamma)
    time_end = perf_counter()
    print('Time for step %d: ' % (i + 1), time_end - time_start)
    time += (time_end - time_start)

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

    crime_gdf = gpd.GeoDataFrame(location, geometry=gpd.points_from_xy(location['LONGITUDE'], location['LATITUDE']))
    obs_indices = np.in1d(arr.reshape(-1), np.array(obs_hist))
    obs_intensity = mean[np.array(obs_hist).astype('int') * grid_value_ratio]
    crime_gdf['intensity'][obs_indices] = obs_intensity

    dc_map = gpd.read_file('../map/washington_dc_boundary.geojson')
    dc_map_block = gpd.read_file('../map/washington_dc_blocks.geojson')
    # fig_dc = plt.figure(figsize=(3, 3))
    # axs_dc = plt.subplot(111, projection=gcrs.Mercator())
    axs_dc = gplt.polyplot(dc_map, projection=gcrs.AlbersEqualArea(), figsize=(3, 3), zorder=1, linewidth=0.7, alpha=0.5)
    gplt.kdeplot(crime_gdf[crime_gdf['intensity'] / max(crime_gdf['intensity']) > 0.75], projection=gcrs.AlbersEqualArea(),
                 ax=axs_dc, cmap='Reds', fill=True, thresh=0, levels=100, legend=True, cut=6, clip=None, alpha=1)
    gplt.polyplot(dc_map_block, projection=gcrs.AlbersEqualArea(), ax=axs_dc, zorder=1, linewidth=0.7, alpha=0.5)
    gplt.pointplot(crime_gdf[crime_gdf['intensity'] / max(crime_gdf['intensity']) > 0.25], projection=gcrs.AlbersEqualArea(),
                   ax=axs_dc, hue='intensity', cmap='Reds', s=0, legend=True)
    gplt.pointplot(crime_gdf[obs_indices], ax=axs_dc, marker='.', color='k', s=12, edgecolor='lightgray', linewidth=1)
    axs_dc.set_xlim(-9100, 8900)
    axs_dc.set_ylim(-1.2e4, 1.2e4)
    plt.show()
    # axs_dc.figure.savefig(save_dir_map + '/step_%d.png' % (i + 1), format='png', bbox_inches='tight', dpi=300)
    # np.save(save_dir + '/center.npy', region_center)
    axs_dc.clear()

print('Total time: ', time)
# np.save(save_dir + '/time.npy', np.array(time))