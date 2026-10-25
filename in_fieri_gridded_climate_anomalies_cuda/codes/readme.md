# Climate Data Processing & Spatial Analytics Pipeline

High-performance spatial climate data processing, moving threshold generation, extreme weather/stress breach detection, and cartographic visualization using **Xarray**, **Zarr**, and **GPU acceleration (CuPy)**.

---

## Technical Stack & Dependencies

- **Core Arrays & I/O:** `xarray`, `numpy`, `pandas`, `zarr`
- **GPU Acceleration:** `cupy`, `cupy_xarray`
- **Geospatial & Plotting:** `cartopy`, `matplotlib`, `rioxarray`
- **Utilities:** `tqdm`, `glob`, `gc`, `pickle`, `datetime`

---

## File & Script Directory

### 1. `climate_calculations.py`
*High-performance spatial processing engine, baseline analytics, and GPU threshold operations.*

* **`CLASS: assemble_zarr_batch`**
  * **Purpose:** Converts raw daily NetCDF climate files into compressed, optimized Zarr stores in manageable batches to handle large-scale spatial datasets.
  * **Methods:**
    * `__init__(netcdfs_path, zarr_batch_size, output_path)`: Initializes batch processing rules and sets up output directories.
    * `make_zarr_batches(area_str, metric_str, metric_rename, interpolate, ...)`: Iterates through raw NetCDF files, renames target climate variables, applies linear temporal interpolation for missing values, and saves consolidated Zarr stores.

* **`CLASS: zarr_grid_descriptives`**
  * **Purpose:** Primary analytical engine for calculating spatial climatologies, moving statistical baselines, extreme thresholds, and breach metrics.
  * **Methods:**
    * `get_daily_zarr_grids(zarr_path, combine_zarrs)`: Loads daily Zarr datasets either combined via metadata or as single-store references.
    * `mean_shifts(time_slices_list, colors)`: Computes bimonthly mean changes (Jan–Feb through Nov–Dec) across multi-year sub-samples.
    * `_make_lower_precipitation_thresholds` / `_make_upper_precipitation_thresholds`: Specialized GPU-accelerated wet-day quantile filters (ignoring zeros).
    * `make_upper_moving_threshold` / `make_lower_moving_threshold`: Calculates calendar-day windowed moving thresholds (using historical windows, gaps, and daily margins) for climate metrics (e.g., VPD, precipitation).
    * `process_and_save_breach_moving_batches(...)`: Compares daily grids against moving thresholds on GPU VRAM, calculates breach occurrence (0/1) and magnitude, and exports compressed NetCDFs.
    * `fixed_breach_shifts(metrics_path, measure_period, time_slices_list, ...)`: Aggregates breach frequency, intensity, or raw count across bimonthly periods and historical time slices.
    * `plot_over_time_changes(header_string, stress_bar_string)`: Generates 6xN multi-panel cartographic figures using Cartopy and standardized colormaps.

* **`FUNCTION: collect_net_cdf`**
  * **Purpose:** Stitches GPU/RAM moving-threshold dictionaries into a unified, continuous Xarray Dataset along the time dimension and exports compressed NetCDF files.

---

### 2. `centralSA_thresholds.py`
*Automates pipeline execution across multi-year observation windows.*

* **Purpose:** Automates raw data extraction, moving threshold generation, and extreme stress breach analysis across multi-year periods (2000–2025).
* **Components:**
  * **Unzipping utility:** Extracts raw precipitation flux archives.
  * **`CLASS: the_thresholds`:** Wraps threshold generation pipelines for both upper and lower quantiles across rolling historical windows (for `prec_flux` and `vpd_max`).
  * **Execution blocks:** Runs parallelized batch processing for extreme upper/lower breaches and magnitude exports.

---

### 3. `centralSA_breaches.py`
*Regional climate exposure and stress time series processing.*

* **Purpose:** Computes regional climate exposure and stress time series for agricultural modeling (e.g., Arabica
