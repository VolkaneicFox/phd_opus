# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob, xarray, os, tqdm, zarr,pickle,gc
import  cupy_xarray, cupy
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import numpy as np

class assemble_zarr_batch:
    def __init__(self,netcdfs_path="",
                 zarr_batch_size=1000,
                 output_path=""):
        
        self.netcdfs_path = sorted(glob.glob(netcdfs_path))
        self.output_path = output_path
        self.zarr_batch_size = zarr_batch_size
        
        os.makedirs(self.output_path,exist_ok=True)
        
        self.total_files = len(self.netcdfs_path)
        
    
    def make_zarr_batches(self,area_str,metric_str,
                          metric_rename,
                          interpolate=True,
                          interpolate_dimension="time",
                          interpolate_method="linear"):
        self.metric_rename=metric_rename
        for i in tqdm.tqdm(range(0, self.total_files, self.zarr_batch_size)):
            batch_paths = self.netcdfs_path[i:i + self.zarr_batch_size]
            batch_num = i // self.zarr_batch_size + 1
            out_file = f"{self.output_path}/{area_str}_{metric_str}_batch_{batch_num:02d}.nc"
            if os.path.exists(out_file):
                print(f"Batch {batch_num} already processed. Skipping...")
                continue
        
            print(f"Processing Batch {batch_num} ({len(batch_paths)} files)...")

            self.nehm_ds = xarray.open_mfdataset(batch_paths, engine="netcdf4",
                                                 combine="by_coords", parallel=True)
            long_name = list(self.nehm_ds.variables.keys())[0]
            print(f"Variable Long Name is = {long_name}")
            self.nehm_ds = self.nehm_ds.rename({long_name: self.metric_rename})
            self.nehm_ds = self.nehm_ds.load()
            if interpolate:
                self.nehm_ds[self.metric_rename] = self.nehm_ds[self.metric_rename].interpolate_na(dim=interpolate_dimension, 
                                                                                                    method=interpolate_method)
            self.nehm_ds.to_zarr(f"{self.output_path}/{area_str}_{metric_str}_batch_{batch_num:02d}.zarr", 
                                  mode="w",consolidated=True)

class zarr_grid_descriptives:
    def __init__(self,metric_str,area_str):
        
        self.metric_str = metric_str
        self.area_str = area_str
        
    def get_daily_zarr_grids(self,zarr_path,combine_zarrs=True):
        self.zarr_path = zarr_path
        if combine_zarrs:
            self.main_zarr=xarray.open_mfdataset(self.zarr_path, 
                                                 combine="by_coords",
                                                 parallel=True,engine="zarr")
        else:
            self.main_zarr=xarray.open_zarr(self.zarr_path)
    
    """
    the descriptive statistics is about looking at how the seasons have changed
    over the period, so we take two months together: Jan/Feb, Mar/Apr ... Nov/Dec
    and take different subsample time slices, and then showcase them together to
    see if there has been an observable change in the VPD. 
    This can be done as 6 rows with month couples, and four to five columns with
    the year slices.
    Four main things:  1. mean shift,  2. fluctuations, 
                       3. local tails, 4. local extremes
    """
    
    """chapter 1. mean shifts"""
    def mean_shifts(self,time_slices_list=[("1990-01-01","1995-12-31"),
                                           ("1996-01-01","2000-12-31"),
                                           ("2001-01-01","2005-12-31"),
                                           ("2006-01-01","2010-12-31"),
                                           ("2011-01-01","2015-12-31"),
                                           ("2016-01-01","2021-12-31"),
                                           ("2022-01-01","2025-12-31")],
                    colors = ["#74c476", "#ffffff", "#de2d26"]):
        
        self.metric_cmap = LinearSegmentedColormap.from_list("cmap_to_give", colors)
        self.time_slices_list=time_slices_list
        self.mean_zarray=self.main_zarr.copy()
        """
        ##month pair indexing explained
        ##because I am a bit of an idiot
        2: 1;   3: 1;    4: 2;    5: 2;    6: 3;    7: 3;   
        8: 4;   9: 4;   10: 5;   11: 5;   12: 6;   13: 6; '
        """
        self.mean_zarray["month_pair"]=(self.mean_zarray.time.dt.month+1)//2
        self.mean_zarray=self.mean_zarray.set_coords("month_pair")
        lons = self.mean_zarray.lon.values
        lats = self.mean_zarray.lat.values
        lon_min, lon_max = lons.min(), lons.max()
        lat_min, lat_max = lats.min(), lats.max()
        self.extent = [lon_min, lon_max, lat_min, lat_max]
        self.origin = "upper" if lats[0] > lats[-1] else "lower"
        # Compute grouping and global scale bounds across slices
        self.computed_slices = []
        self.global_min, self.global_max = float("inf"), float("-inf")
        ##in the figure, the rows are the 6 month-pairs, and the columns
        ##are the time slices
        
        for time_slice in time_slices_list:
            start_time,end_time=pd.to_datetime(time_slice[0]),pd.to_datetime(time_slice[1])
            sliced_zarray = self.mean_zarray.sel(time=slice(start_time,end_time))

            mean_sliced_zarray=sliced_zarray.groupby("month_pair").mean(dim="time")
            self.mean_sliced_zarray = mean_sliced_zarray[self.metric_str].compute()

            self.global_min = min(self.global_min, float(self.mean_sliced_zarray.min()))
            self.global_max = max(self.global_max, float(self.mean_sliced_zarray.max()))
    
            self.computed_slices.append((time_slice[0][:4], time_slice[1][:4], self.mean_sliced_zarray))
    
    
    """
    fixed thresholds are in netcdf_explore.py
    chapter 2. local tails against fixed historical margins
    
    in a leap year, gather all dates, then make it a clock, for each date we have
    +/- number of margin days, and then string concat with the years to be used as
    historical reference
    
    chapter 3. local tails against moving historical margins
        
    each year has its batch of reference years, starting with 5 years prior, so
    the year 2000, has the reference batch 1990-95,  and the year  2001 has the
    reference batch 1991-96, and so on.

    therefore, for each year we slice the relevant history and repeat the fixed
    batch mutations    

    """

    
    def _make_lower_precipitation_thresholds(self, box, lower_quantile):
        """Calculates the quantile (e.g., 0.05) strictly over wet days (box > 0)
        """
        wet_vals = cupy.where(box > 0, box, cupy.nan)
        n_wet = cupy.sum(box > 0, axis=0)
        srt = cupy.sort(wet_vals, axis=0)
        pos = (n_wet * lower_quantile).astype(cupy.int64)
        pos = cupy.clip(pos, 0, cupy.maximum(n_wet - 1, 0))
        low_thr = cupy.take_along_axis(srt, pos[None, ...], axis=0)[0]
        low_thr = cupy.where(n_wet > 0, low_thr,0)
        return low_thr
    
    def _make_upper_precipitation_thresholds(self,box,upper_quantile):
        wet_vals = cupy.where(box > 0, box, cupy.nan)
        n_wet = cupy.sum(box > 0, axis=0)
        srt = cupy.sort(wet_vals, axis=0)
        pos = (n_wet * upper_quantile).astype(cupy.int64)
        pos = cupy.clip(pos, 0, cupy.maximum(n_wet - 1, 0))
        up_thr = cupy.take_along_axis(srt, pos[None, ...], axis=0)[0]
        up_thr = cupy.where(n_wet > 0, up_thr,0)
        return up_thr
    
    
    def _make_upper_margin_threshold(self,upper_quantile,present_year,
                                     print_process_time=True):
        hist_times = pd.to_datetime(self.hist_zarr.time.values)
        full_mmdd_series = hist_times.strftime("%m-%d").values

        self.unique_calendar_days = np.unique(full_mmdd_series)
        hist_gpu = cupy.asarray(self.hist_zarr.values, dtype=cupy.float32)
        n_total_days = hist_gpu.shape[0]

        self.up_thresholds = {}
        start_time=datetime.now().strftime("%H:%M:%S")
        mempool = cupy.get_default_memory_pool()
        for target_day in self.unique_calendar_days:
            center_indices = np.where(full_mmdd_series == target_day)[0]
            window_indices = []
            for center_idx in center_indices:
                win = np.arange(center_idx - self.margin_of_days, center_idx + self.margin_of_days + 1)
                win = win % n_total_days
                window_indices.extend(win)
            pooled_gpu_indices = cupy.array(window_indices)
            pooled_sample = hist_gpu[pooled_gpu_indices]
            if "prec" in self.metric_str:
                up_q_gpu=self._make_upper_precipitation_thresholds(box=pooled_sample,
                                                            upper_quantile=upper_quantile)
            else:
                up_q_gpu = cupy.quantile(pooled_sample, q=upper_quantile, axis=0)
            self.up_thresholds[f"{present_year}-{target_day}"] = up_q_gpu.get().astype(np.float32)
            del pooled_gpu_indices, pooled_sample, up_q_gpu, 
            mempool.free_all_blocks()
            
        del hist_gpu
        mempool.free_all_blocks()
        gc.collect()   
        print(f"Done! Generated moving upper thresholds for {len(self.unique_calendar_days)} days in year {present_year}.")
        end_time=datetime.now().strftime("%H:%M:%S")
        if print_process_time:
            print("start_time = ",start_time)
            print("end time   = ",end_time)
        else:
            None
            
    def _make_lower_margin_threshold(self,lower_quantile,present_year,
                                     print_process_time=True):
        hist_times = pd.to_datetime(self.hist_zarr.time.values)
        full_mmdd_series = hist_times.strftime("%m-%d").values

        self.unique_calendar_days = np.unique(full_mmdd_series)
        hist_gpu = cupy.asarray(self.hist_zarr.values, dtype=cupy.float32)
        n_total_days = hist_gpu.shape[0]

        self.low_thresholds = {}
        start_time=datetime.now().strftime("%H:%M:%S")
        mempool = cupy.get_default_memory_pool()
        for target_day in self.unique_calendar_days:
            center_indices = np.where(full_mmdd_series == target_day)[0]
            window_indices = []
            for center_idx in center_indices:
                win = np.arange(center_idx - self.margin_of_days, center_idx + self.margin_of_days + 1)
                win = win % n_total_days
                window_indices.extend(win)
            pooled_gpu_indices = cupy.array(window_indices)
            pooled_sample = hist_gpu[pooled_gpu_indices]
            if "prec" in self.metric_str:
                low_q_gpu =  self._make_lower_precipitation_thresholds(box=pooled_sample,
                                                             lower_quantile=lower_quantile)
            else:
                low_q_gpu = cupy.quantile(pooled_sample, q=lower_quantile, axis=0)
            self.low_thresholds[f"{present_year}-{target_day}"] = low_q_gpu.get().astype(np.float32)
                
            del pooled_gpu_indices, pooled_sample
            mempool.free_all_blocks()
            
        del hist_gpu
        mempool.free_all_blocks()
        gc.collect()   
        print(f"Done! Generated moving lower thresholds for {len(self.unique_calendar_days)} days in year {present_year}.")
        end_time=datetime.now().strftime("%H:%M:%S")
        if print_process_time:
            print("start_time = ",start_time)
            print("end time   = ",end_time)
        else:
            None
        
    def make_upper_moving_threshold(self,present_start_date,present_end_date,
                               hist_delta=5,margin_of_days=5,hist_gap=5,
                               upper_quantile=0.95):
        self.present_moving_start_date=present_start_date
        self.present_moving_end_date  = present_end_date
        self.hist_delta=hist_delta
        self.hist_gap=hist_gap
        self.margin_of_days=margin_of_days
        self.now_zarr = self.main_zarr.sel(time=slice(self.present_moving_start_date,
                                                      self.present_moving_end_date))
        self.present_years = np.arange(int(self.present_moving_start_date.split('-')[0]),
                               int(self.present_moving_end_date.split('-')[0])+1,
                               1)
        start_time=datetime.now().strftime("%H:%M:%S")
        self.moving_upper_thresholds=[]
        for present_year in self.present_years:
            hist_end_year = present_year - self.hist_gap
            hist_start_year = hist_end_year - self.hist_delta
            hist_start_date_str,hist_end_date_str = f"{hist_start_year}-01-01",f"{hist_end_year}-12-31"
            
            self.hist_zarr = self.main_zarr[self.metric_str].sel(time=slice(hist_start_date_str,
                                                                  hist_end_date_str)).load()
            self._make_upper_margin_threshold(upper_quantile=upper_quantile, 
                                              present_year=present_year)
            self.moving_upper_thresholds.append(self.up_thresholds)
            gc.collect()
        end_time=datetime.now().strftime("%H:%M:%S")
        
        print(f"Time taken is from {start_time} to {end_time}")
            

            
    def make_lower_moving_threshold(self,present_start_date,present_end_date,
                               hist_delta=5,margin_of_days=5,hist_gap=5,
                               lower_quantile=0.05,print_process_time=True):
        self.present_moving_start_date=present_start_date
        self.present_moving_end_date  = present_end_date
        self.hist_delta=hist_delta
        self.hist_gap=hist_gap
        self.margin_of_days=margin_of_days
        self.now_zarr = self.main_zarr.sel(time=slice(self.present_moving_start_date,
                                                      self.present_moving_end_date))
        self.present_years = np.arange(int(self.present_moving_start_date.split('-')[0]),
                               int(self.present_moving_end_date.split('-')[0])+1,
                               1)
        start_time=datetime.now().strftime("%H:%M:%S")
        self.moving_lower_thresholds=[]
        for present_year in self.present_years:
            hist_end_year = present_year - self.hist_gap
            hist_start_year = hist_end_year - self.hist_delta
            hist_start_date_str,hist_end_date_str = f"{hist_start_year}-01-01",f"{hist_end_year}-12-31"
            
            self.hist_zarr = self.main_zarr[self.metric_str].sel(time=slice(hist_start_date_str,
                                                                  hist_end_date_str)).load()
            self._make_lower_margin_threshold(lower_quantile=lower_quantile, 
                                              present_year=present_year)
            self.moving_lower_thresholds.append(self.low_thresholds)
            gc.collect()
        print(f"Done! Generated moving lower thresholds for {len(self.unique_calendar_days)} days in year {present_year}.")
        end_time=datetime.now().strftime("%H:%M:%S")
        if print_process_time:
            print("start_time = ",start_time)
            print("end time   = ",end_time)
        else:
            None
            
        
    def process_and_save_breach_moving_batches(self, present_start_date, present_end_date,
                                               moving_thresholds_dir,thresh_type="upper",
                                        output_dir="./breach_batches", batch_size=1000):
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.present_end_date =present_end_date
        self.present_start_date=present_start_date
    
        self.moving_thresholds=xarray.open_mfdataset(paths=moving_thresholds_dir)
        self.moving_thresholds = self.moving_thresholds.sortby("time").sel(
            time=slice(self.present_start_date,self.present_end_date))
        self.now_zarr = self.main_zarr[self.metric_str].sortby("time").sel(
            time=slice(self.present_start_date, self.present_end_date))
        
        t_data = self.now_zarr.time.values
        t_thr = self.moving_thresholds.time.values
        if t_data.size == 0 or not np.array_equal(t_data, t_thr):
            raise ValueError(f"Time stamps are not aligned: data {t_data.size} days, thresholds {t_thr.size} days")
        print("Dates work")
        now_times = pd.to_datetime(self.now_zarr.time.values)
        lats = self.now_zarr.lat.values
        lons = self.now_zarr.lon.values
        n_total_days = len(now_times)
        
        mempool = cupy.get_default_memory_pool()
        
        for batch_num, start_idx in enumerate(range(0, n_total_days, batch_size)):
            end_idx = min(start_idx + batch_size, n_total_days)
            batch_times = now_times[start_idx:end_idx]
            
            batch_xr = self.now_zarr.isel(time=slice(start_idx, end_idx))
            batch_gpu = cupy.asarray(batch_xr.values, dtype=cupy.float32)
            
            thresh_xr = self.moving_thresholds[thresh_type].isel(time=slice(start_idx,end_idx))
            thresh_gpu = cupy.asarray(thresh_xr.values, dtype=cupy.float32)

            if thresh_type == "upper":
                breach_gpu = batch_gpu > thresh_gpu                     
                breach_gpu_mag = cupy.where(breach_gpu, batch_gpu - thresh_gpu, 0.0)
            elif thresh_type == "lower":
                breach_gpu = batch_gpu < thresh_gpu
                breach_gpu_mag = cupy.where(breach_gpu, thresh_gpu - batch_gpu, 0.0)
            else:
                raise ValueError(f"thresh_type must be 'upper' or 'lower', got {thresh_type}")
            breach_cpu = breach_gpu.get().astype(np.uint8)
            breach_cpu_mag = breach_gpu_mag.get().astype(np.float32)
            test_neg_breach=breach_cpu_mag<0
            if cupy.sum(test_neg_breach).item()>0:
                print(batch_times[0])
                print(batch_times[-1])
                raise ValueError("We have negative breaches!")
            
            ds_batch = xarray.Dataset(
                data_vars={
                    f"{thresh_type}_breach": (["time", "lat", "lon"], breach_cpu),
                    
                },
                coords={"time": batch_times, "lat": lats, "lon": lons},
                attrs={'thresh':thresh_type,
                    "metric_str": self.metric_str
                }
            )
            file_name = f"{thresh_type}breaches_batch_{batch_num:03d}_{batch_times[0].strftime('%Y%m%d')}_{batch_times[-1].strftime('%Y%m%d')}.nc"
            ds_batch.to_netcdf(os.path.join(output_dir, file_name), encoding={
                f"{thresh_type}_breach": {"zlib": True, "complevel": 4}
            })
            
            ds_batch_mag = xarray.Dataset(
                data_vars={
                    f"{thresh_type}_breach_mag": (["time", "lat", "lon"], breach_cpu_mag)
                },
                coords={"time": batch_times, "lat": lats, "lon": lons},
                attrs={"thresh":thresh_type, "metric_str":self.metric_str}
            )
            file_name_mag = f"{thresh_type}breachmag_batch_{batch_num:03d}_{batch_times[0].strftime('%Y%m%d')}_{batch_times[-1].strftime('%Y%m%d')}.nc"
            ds_batch_mag.to_netcdf(os.path.join(output_dir, file_name_mag), encoding={
                f"{thresh_type}_breach_mag": {"zlib": True, "complevel": 4}
            })
    
            del batch_gpu, thresh_gpu, breach_gpu, breach_gpu_mag
            mempool.free_all_blocks()
            
            print(f"Saved batch {batch_num:03d} -> {file_name}")
    
        print(f"\nAll batches saved to '{output_dir}'.")
    
    
    def fixed_breach_shifts(self, metrics_path="", measure_period="mean",
                        time_slices_list=[("2001-01-01","2005-12-31"),
                                          ("2006-01-01","2010-12-31"),
                                          ("2011-01-01","2015-12-31"),
                                          ("2016-01-01","2021-12-31"),
                                          ("2022-01-01","2025-12-31")],
                        metric_str="",
                        colors=["#74c476", "#ffffff", "#de2d26"]):
        self.metrics_xr = xarray.open_mfdataset(paths=metrics_path)

        try:
            metric_str = list(self.metrics_xr.data_vars)[0]
        except:
            metric_str = metric_str
        
        self.metric_cmap = LinearSegmentedColormap.from_list("cmap_to_give", colors)
        self.time_slices_list = time_slices_list
        
        lons = self.metrics_xr.lon.values
        lats = self.metrics_xr.lat.values
        self.extent = [lons.min(), lons.max(), lats.min(), lats.max()]
        self.origin = "upper" if lats[0] > lats[-1] else "lower"
        
        self.computed_slices = []
        self.global_min, self.global_max = float("inf"), float("-inf")
        
        # Process each time slice
        for time_slice in time_slices_list:
            start_time, end_time = pd.to_datetime(time_slice[0]), pd.to_datetime(time_slice[1])
            sliced_zarray = self.metrics_xr[metric_str].sel(time=slice(start_time, end_time))
            sliced_times = pd.to_datetime(sliced_zarray.time.values)
            
            sliced_cupy = cupy.asarray(sliced_zarray.values, dtype=cupy.float32)

            slice_month_pairs = ((sliced_times.month + 1) // 2)
            
            bimonthly_grids = {}

            for mm_pair in range(1, 7):
                pair_mask = (slice_month_pairs == mm_pair)
                pair_indices_gpu = cupy.array(np.where(pair_mask)[0])
                
                pair_data_gpu = sliced_cupy[pair_indices_gpu, :, :]
                
                # Aggregation: Mean or Sum
                # Select metric calculation based on user option
                if measure_period == "frequency":
                    grid_gpu = (pair_data_gpu > 0).mean(axis=0)
                
                elif measure_period == "intensity":
                    counts = (pair_data_gpu > 0).sum(axis=0)
                    sums = pair_data_gpu.sum(axis=0)
                    grid_gpu = cupy.where(counts > 0, sums / counts, 0.0)
                
                elif measure_period == "count":
                    grid_gpu = (pair_data_gpu > 0).sum(axis=0)
                
                else:
                    grid_gpu = (pair_data_gpu > 0).mean(axis=0)
                                
                # Update global min/max for standardized colormap scaling
                self.global_min = min(self.global_min, float(grid_gpu.min()))
                self.global_max = max(self.global_max, float(grid_gpu.max()))

                bimonthly_grids[mm_pair] = grid_gpu
                
            # Store tuple: (start_year, end_year, {1: grid_p1, 2: grid_p2, ..., 6: grid_p6})
            slice_label = f"{time_slice[0][:4]}-{time_slice[1][:4]}"
            self.computed_slices.append((slice_label, bimonthly_grids))

            del sliced_cupy
            cupy.get_default_memory_pool().free_all_blocks()
    
        print(f"Processed {len(self.computed_slices)} time slices.")
        print(f"Global min: {self.global_min:.4f}, Global max: {self.global_max:.4f}")
    
    def _plot_single_panel(self, ax, grid_2d, projection, is_first_row, 

                          is_first_col, col_title, row_label):
        if hasattr(grid_2d, "get"):
           grid_2d = grid_2d.get()

        ax.set_extent(self.extent, crs=projection)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#222222")
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="#333333")
        ax.add_feature(self.provincial_lines, linewidth=0.3, edgecolor="#666666", linestyle=":")

        im = ax.imshow(grid_2d, cmap=self.metric_cmap, vmin=self.global_min,
                      vmax=self.global_max, extent=self.extent,
                      origin=self.origin, transform=projection)
       
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        if is_first_row:
           ax.set_title(col_title, fontsize=11, fontweight="bold", pad=8)
        if is_first_col:
            ax.text(-0.08, 0.5, row_label, transform=ax.transAxes,
                   fontsize=10, fontweight="bold", va="center", ha="right")
        return im 
    
    
    def plot_over_time_changes(self, header_string="", stress_bar_string=""):
        self.provincial_lines = cfeature.NaturalEarthFeature(
            category="cultural",
            name="admin_1_states_provinces_lines",
            scale="50m",
            facecolor="none"
        )
        n_cols = len(self.time_slices_list)
        projection = ccrs.PlateCarree()
    
        # 4. Removed sharex/sharey to prevent Cartopy GeoAxes conflicts
        fig, axes = plt.subplots(nrows=6, ncols=n_cols,
                                 figsize=(n_cols * 2.8, 12),
                                 subplot_kw={"projection": projection})
    
        month_pair_labels = ["Jan-Feb", "Mar-Apr", "May-Jun",
                             "Jul-Aug", "Sep-Oct", "Nov-Dec"]
    
        # Populate grid panels using helper function
        last_im = None
        for col_idx, (slice_label, bimonthly_grids) in enumerate(self.computed_slices):
            for row_idx, mm_pair in enumerate(range(1, 7)):
                
                # Support both 1D and 2D axes indexing if n_cols == 1
                ax = axes[row_idx, col_idx] if n_cols > 1 else axes[row_idx]
                grid_2d = bimonthly_grids[mm_pair]
    
                last_im = self._plot_single_panel(
                    ax=ax,
                    grid_2d=grid_2d,
                    projection=projection,
                    is_first_row=(row_idx == 0),
                    is_first_col=(col_idx == 0),
                    col_title=slice_label,
                    row_label=month_pair_labels[row_idx]
                )
                
        fig.suptitle(header_string, fontsize=14, fontweight="bold", y=0.96)
    
        # Horizontal shared colorbar at bottom
        cbar_ax = fig.add_axes([0.30, 0.04, 0.40, 0.015])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
        cbar.set_label(stress_bar_string, fontsize=10, fontweight="bold")
        cbar.ax.tick_params(labelsize=9)
    
        plt.subplots_adjust(top=0.91, bottom=0.08, left=0.10, right=0.98,
                            hspace=0.08, wspace=0.08)
    
        self.mean_fig = fig
        return fig
    
    
def collect_net_cdf(list_of_gis,thresh_type,
                    lats,lons,metric_str,area_str,
                    description_str,hist_delta,
                    hist_gap,margin_of_days,
                    window,output_path):
    """
    Merges all year-wise moving threshold dictionaries into a single continuous 
    Xarray Dataset with a proper 'time' dimension and saves it to NetCDF.
    """
    master_up = {}
    for d in list_of_gis:
        master_up.update(d)
    sorted_dates=[]   
    # sort date keys chronologically (e.g., "2000-01-01", "2000-01-02", ...)
    for date in sorted(list(master_up.keys())):
        try:
            sorted_dates.append(pd.to_datetime(date))
        except:
            None
    
    if not sorted_dates:
        print("Error: No moving thresholds found in memory to save!")
        return
        
    print(f"Stacking {len(sorted_dates)} daily moving threshold grids into VRAM tensor...")

    up_stacked_gpu = cupy.stack([cupy.asarray(master_up[str(date_str).split(" ")[0]]) for date_str in sorted_dates], axis=0)

    up_cpu = up_stacked_gpu.get().astype(np.float32)

    

    ds_thresholds = xarray.Dataset(
        data_vars={
           thresh_type : (["time", "lat", "lon"], up_cpu),
        },
        coords={
            "time": sorted_dates,
            "lat": lats,
            "lon": lons
        },
        attrs={
            "description": description_str,
            "metric_str": metric_str,
            "hist_delta_years": hist_delta,
            "hist_gap_years": hist_gap,
            "margin_of_days": margin_of_days,
            "reference_window": window,
            "area": area_str
        }
    )
    encoding = {
        thresh_type: {"zlib": True, "complevel": 4}
    }
    
    ds_thresholds.to_netcdf(output_path, encoding=encoding)
    print("Master moving thresholds successfully saved")
    
    # Clean up GPU memory
    del up_stacked_gpu
    cupy.get_default_memory_pool().free_all_blocks()
    
    return ds_thresholds

