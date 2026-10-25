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

* **Purpose:** Computes regional climate exposure and stress time series for agricultural modeling (e.g., Arabica coffee regions in Central South America / Brazil).
* **Components:**
  * `make_climate_compressed_timeseries`: Aligns climate datasets across temporal phases (2005–09, 2010–20, 2020–25) using MapSPAM spatial weights.
  * **Precipitation Flux processing:** Calculates upper breach extent/magnitude (normalized against climatological thresholds, handling zero-bounds) and lower breach count shares.
  * **VPD Max processing:** Extracts upper stress breach extents and shares.
  * **Daily gridded means:** Computes and exports raw daily mean values for econometric inputs.
  * **CSV Export:** Outputs structured regional indicator time series to `timeseries_csv/`.

---

### 4. `grid_compression.py`
*Spatial weighting and crop area spatial partitioning.*

* **Purpose:** Processes MapSPAM physical crop distribution raster data (e.g., Arabica coffee) across historical periods, divides regions into four spatial quadrants, computes normalized area weights, and extracts spatially-weighted regional climate time series.
* **Components:**
  * **`CLASS: the_growing_climate_areas`:** Loads raster GeoTIFFs, partitions spatial extents into quadrants, normalizes physical area weights (summing to 1.0), and exports NetCDF weight grids.
  * **`CLASS: weighted_time_series`:** Aligns and merges quadrant weight matrices with climate Zarr datasets to compute weighted regional averages for econometric modeling.
  * `making_arabica_weights`: Automation function for generating Arabica coffee spatial weights across multiple historical periods (2005, 2010, 2020).

---

### 5. `saving_time_series.py`
*Time series extraction and export utility.*

* **Purpose:** Extracts, processes, and exports regional climate stress and daily mean indicators for agricultural modeling (Arabica coffee quadrants) across multiple multi-year phases.
* **Components:**
  * `make_climate_compressed_timeseries`: Aligns climate datasets with MapSPAM spatial weights across defined historical phases (2005–09, 2010–20, 2020–25).
  * **Precipitation Flux extraction:** Computes and exports upper breach extent, upper share, and lower breach count shares to CSV.
  * **VPD Max extraction:** Computes and exports upper breach extent and upper share metrics.
  * **Daily baseline means:** Computes and exports raw daily mean gridded values for both precipitation flux and maximum VPD.
 
*markdown prepared with assistance from Gemini*

* **Purpose:** Computes regional climate exposure and stress time series for agricultural modeling (e.g., Arabica
