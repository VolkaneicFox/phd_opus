#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:25:31 2026

@author: darkmage
"""
from netcdf_explore import zarr_grid_descriptives

zarr_grids_vpd=zarr_grid_descriptives(metric_str="vpd_max", area_str="brazil")

zarr_grids_vpd.get_daily_zarr_grids(zarr_path="/home/darkmage/paper_three/vpd_processed_batches/*.zarr")
zarr_grids_vpd.make_fixed_margin_thresholds(path_fixed_margin_references="/home/darkmage/paper_three/vpd_netcdfs/brazil/fixed_margin_threshold_windows5/",
                                            margin_of_days=5,save_thresholds_path="/home/darkmage/paper_three/vpd_netcdfs/brazil/fixed_1990_96/")
zarr_grids_vpd.process_and_save_breach_fixed_batches(present_start_date="2000-01-01", present_end_date="2025-12-31",
                                               output_dir="/home/darkmage/paper_three/vpd_netcdfs/brazil/fixed_1990_96/breach_of_thresholds/")