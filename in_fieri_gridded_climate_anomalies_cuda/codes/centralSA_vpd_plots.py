#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:00:26 2026

@author: darkmage
"""

from netcdf_explore import zarr_grid_descriptives


# %%

centralSA_vpd = zarr_grid_descriptives(metric_str="vpd_max", area_str="centralSA")
centralSA_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/vpd_processed_batches/*.zarr")
centralSA_vpd.fixed_breach_shifts(metrics_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/movingbreach_window5delta5/breachmag_*.nc",colors=["#FFFFFF", "#de2d26"],
                                  metric_str="vpd_max",measure_period="intensity")
centralSA_vpd.plot_over_time_changes(header_string="VPD Anomalies against Shifting Baseline in Central S. America, 2000-2025",
                                     stress_bar_string="VPD Anomaly X100 %")


# %%% FIXED THRESHOLD VPD BRAZIL            
zarr_grids_vpd=zarr_grid_descriptives(metric_str="vpd_max", area_str="brazil")

zarr_grids_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_processed_batches/*.zarr")
zarr_grids_vpd.fixed_breach_shifts(metrics_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/fixed_1990_96/breach_of_thresholds/breachmag_*.nc",
                                   measure_period="intensity",colors=["#FFFFFF", "#de2d26"])
zarr_grids_vpd.plot_over_time_changes(header_string="VPD Anomalies against 1950-80 Baseline in Central S. America, 2000-2025",stress_bar_string="VPD Anomaly X100 %")
