#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:01:10 2026

@author: darkmage
"""

"""
date batch = 2021-01-01 to 2025-12-31
"""

from netcdf_explore import zarr_grid_descriptives, collect_net_cdf

#%%
centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.fixed_breach_shifts(metrics_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/movingbreach_window5delta5/breachmag_*.nc",colors=["#c0eda3", "#de2d26"],
                                  metric_str="vpd_max",measure_period="intensity")
#%%

t0,t1="2021-01-01","2025-12-31"
y0,y1=t0[2:4],t1[2:4]
centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.make_moving_thresholds(present_start_date=t0, present_end_date=t1,
                                     hist_delta=5,hist_gap=5,margin_of_days=5,lower_quantile=0.05,
                                     upper_quantile=0.95)
upper_threshold=centralSA_vpd.moving_upper_thresholds
thresh=collect_net_cdf(upper_threshold, 
                lats=centralSA_vpd.main_zarr.lat, lons=centralSA_vpd.main_zarr.lon, 
                metric_str="vpd_max", area_str="central SA", 
                description_str=f"Moving daily thresholds from {t0} to {t1}", 
                hist_delta=5, hist_gap=5, margin_of_days=5, window="moving", 
                output_path=f"/home/darkmage/paper_three/vpd_netcdfs/brazil/moving_threshold_window5_delta5/moving_upper_thresholds_{y0}to{y1}.nc")

# %%
"""
date batch = 2016-01-01 to 2020-12-31
"""
t0,t1="2016-01-01","2020-12-31"
y0,y1=t0[2:4],t1[2:4]
centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.make_moving_thresholds(present_start_date=t0, present_end_date=t1,
                                     hist_delta=5,hist_gap=5,margin_of_days=5,lower_quantile=0.05,
                                     upper_quantile=0.95)
upper_threshold=centralSA_vpd.moving_upper_thresholds
thresh=collect_net_cdf(upper_threshold, 
                lats=centralSA_vpd.main_zarr.lat, lons=centralSA_vpd.main_zarr.lon, 
                metric_str="vpd_max", area_str="central SA", 
                description_str=f"Moving daily thresholds from {t0} to {t1}", 
                hist_delta=5, hist_gap=5, margin_of_days=5, window="moving", 
                output_path=f"/home/darkmage/paper_three/vpd_netcdfs/brazil/moving_threshold_window5_delta5/moving_upper_thresholds_{y0}to{y1}.nc")



# %%
"""
date batch = 2011-01-01 to 2015-12-31
"""

t0,t1="2011-01-01","2015-12-31"
y0,y1=t0[2:4],t1[2:4]
centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.make_moving_thresholds(present_start_date=t0, present_end_date=t1,
                                     hist_delta=5,hist_gap=5,margin_of_days=5,lower_quantile=0.05,
                                     upper_quantile=0.95)
upper_threshold=centralSA_vpd.moving_upper_thresholds
thresh=collect_net_cdf(upper_threshold, 
                lats=centralSA_vpd.main_zarr.lat, lons=centralSA_vpd.main_zarr.lon, 
                metric_str="vpd_max", area_str="central SA", 
                description_str=f"Moving daily thresholds from {t0} to {t1}", 
                hist_delta=5, hist_gap=5, margin_of_days=5, window="moving", 
                output_path=f"/home/darkmage/paper_three/vpd_netcdfs/brazil/moving_threshold_window5_delta5/moving_upper_thresholds_{y0}to{y1}.nc")


# %%
"""
date batch = 2006-01-01 to 2010-12-31
"""
centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.make_moving_thresholds(present_start_date="2006-01-01", present_end_date="2010-12-31",
                                     hist_delta=5,hist_gap=5,margin_of_days=5,lower_quantile=0.05,
                                     upper_quantile=0.95)
upper_threshold=centralSA_vpd.moving_upper_thresholds
thresh=collect_net_cdf(upper_threshold, 
                lats=centralSA_vpd.main_zarr.lat, lons=centralSA_vpd.main_zarr.lon, 
                metric_str="vpd_max", area_str="central SA", 
                description_str="Moving daily thresholds from 2006-01-01 to 2010-12-31", 
                hist_delta=5, hist_gap=5, margin_of_days=5, window="moving", 
                output_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/moving_threshold_window5_delta5/moving_upper_thresholds_06to10.nc")




# %%% moving thresholds for VPD in Central South America

"""
date batch = 2000-01-01 to 2005-12-31
"""
centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.make_moving_thresholds(present_start_date="2000-01-01", present_end_date="2005-12-31",
                                     hist_delta=5,hist_gap=5,margin_of_days=5,lower_quantile=0.05,
                                     upper_quantile=0.95)
upper_threshold=centralSA_vpd.moving_upper_thresholds
thresh=collect_net_cdf(upper_threshold, 
                lats=centralSA_vpd.main_zarr.lat, lons=centralSA_vpd.main_zarr.lon, 
                metric_str="vpd_max", area_str="central SA", 
                description_str="Moving daily thresholds from 2000-01-01 to 2005-12-31", 
                hist_delta=5, hist_gap=5, margin_of_days=5, window="moving", 
                output_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/moving_threshold_window5_delta5/moving_upper_thresholds_00to05.nc")