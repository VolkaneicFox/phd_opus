
climate_calculations.py
Overview:
High-performance spatial climate data processing, moving threshold generation, extreme weather/stress breach detection, and cartographic 
visualization using Xarray, Zarr, and GPU acceleration (CuPy).
1. CLASS: assemble_zarr_batch
   Purpose:
   Converts raw daily NetCDF climate files into compressed, optimized Zarr 
   stores in manageable batches to handle large-scale spatial datasets.
   Methods:
   - __init__(netcdfs_path, zarr_batch_size, output_path):
     Initializes batch processing rules and sets up output directories.
   - make_zarr_batches(area_str, metric_str, metric_rename, interpolate, ...):
     Iterates through raw NetCDF files, renames target climate variables, 
     applies linear temporal interpolation for missing values, and saves 
     consolidated Zarr stores.
2. CLASS: zarr_grid_descriptives
   Purpose:
   Primary analytical engine for calculating spatial climatologies, moving 
   statistical baselines, extreme thresholds, and breach metrics.
   Methods:
   - get_daily_zarr_grids(zarr_path, combine_zarrs):
     Loads daily Zarr datasets either combined via metadata or as single-store 
     references.
   - mean_shifts(time_slices_list, colors):
     Computes bimonthly mean changes (Jan-Feb through Nov-Dec) across multi-year 
     sub-samples.
   - _make_lower_precipitation_thresholds / _make_upper_precipitation_thresholds:
     Specialized GPU-accelerated wet-day quantile filters (ignoring zeros).
   - make_upper_moving_threshold / make_lower_moving_threshold:
     Calculates calendar-day windowed moving thresholds (using historical windows, 
     gaps, and daily margins) for climate metrics (e.g., VPD, precipitation).
   - process_and_save_breach_moving_batches(...):
     Compares daily grids against moving thresholds on GPU VRAM, calculates 
     breach occurrence (0/1) and magnitude, and exports compressed NetCDFs.
   - fixed_breach_shifts(metrics_path, measure_period, time_slices_list, ...):
     Aggregates breach frequency, intensity, or raw count across bimonthly 
     periods and historical time slices.
   - plot_over_time_changes(header_string, stress_bar_string):
     Generates 6xN multi-panel cartographic figures using Cartopy and 
     standardized colormaps.
3. FUNCTION: collect_net_cdf
   Purpose:
   Stitches GPU/RAM moving-threshold dictionaries into a unified, continuous 
   Xarray Dataset along the time dimension and exports compressed NetCDF files.
TECHNICAL STACK & DEPENDENCIES
- Core Arrays & I/O : xarray, numpy, pandas, zarr
- GPU Acceleration   : cupy, cupy_xarray
- Geospatial/Plotting: cartopy, matplotlib
- Utilities          : tqdm, glob, gc, pickle, datetime

