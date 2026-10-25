

#%%
drive_path="/media/darkmage/SSD-Storage1/"
hist_delta=5
margin_of_days=5
import xarray as xr 
#import numpy as  np
#import cupy as cp
import pandas as pd

from climate_calculations import zarr_grid_descriptives

"""
will drop 29 Feb Calendar Dates
"""
from grid_compression import weighted_time_series
def make_climate_compressed_timeseries(crop,climate_df):
    phase_periods={"2005to09":("2005-01-01","2009-12-31"),
                  "2010to20":("2010-01-01","2019-12-31"),
                  "2020to25":("2020-01-01","2025-12-31")}
    
    df_dict={}
    grids_weight_path=drive_path+"mapspam_grids"
    
    for phase in phase_periods:
        t0,t1=phase_periods[phase][0],phase_periods[phase][1]
    
        time_series=weighted_time_series(crop=crop, 
                                         period=phase, 
                                         grids_weights_path=grids_weight_path)
        climate_zarr=climate_df.sel(time=slice(t0,t1))
        time_series.align_the_dimensions(climate_zarr=climate_zarr)
        df_dict[phase]=time_series.save_the_time_series()
    return df_dict

#%%
metric_str="prec_flux"
zarr_path=f"{drive_path}/prec_flux/brazil/prec_zarr_processed/*.zarr"

main_call=zarr_grid_descriptives(metric_str=metric_str, area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)
main_zarr=main_call.main_zarr
take_region=main_zarr.notnull().any("time")
take_region=take_region.load() 
#take_region["prec_flux"].plot()
take_region = take_region[metric_str]    
#%% prec flux extent of breach over grids in quadrants
thresh_type='upper'
threshold_path=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_*.nc"
upperbreaches = xr.open_mfdataset(f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/upperbreach*mag*.nc", combine="by_coords")
upperthresh = xr.open_mfdataset(threshold_path)

upperbreaches = upperbreaches["upper_breach_mag"].where(take_region)
upperthresh = upperthresh["upper"].where(take_region)

upperthresh = xr.where(upperthresh == 0, 0.01, upperthresh)   
upperextent = (upperbreaches / upperthresh).rename("upper_extent")
upperextent=upperextent.to_dataset()

crop="arabica"
prec_flux_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=upperextent,thresh_type="upper")
dfs_list=[]
for item in prec_flux_upper_dict:
    dfs_list.append(prec_flux_upper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df

prec_flux_upper_df.to_csv(drive_path+"/timeseries_csv/prec_flux_upperextent.csv")

prec_flux_upper_df=pd.read_csv(drive_path+"/timeseries_csv/prec_flux_upperextent.csv")
prec_flux_upper_df=prec_flux_upper_df.set_index("Unnamed: 0")
prec_flux_upper_df=prec_flux_upper_df.add_prefix(f"{metric_str}_")
prec_flux_upper_df.to_csv(drive_path+"/timeseries_csv/prec_flux_upperextent.csv")
#%% prec_flux upper breaches share of the grids in each quadrant
upperbreaches = xr.open_mfdataset(f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/upperbreaches*.nc", combine="by_coords")
thresh_type="upper"
metric_str="prec_flux"
upperbreaches = upperbreaches["upper_breach"].where(take_region)
upperbreaches=upperbreaches.to_dataset()
crop="arabica"
prec_flux_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=upperbreaches,thresh_type=thresh_type)
dfs_list=[]
for item in prec_flux_upper_dict:
    dfs_list.append(prec_flux_upper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df=prec_flux_upper_df.add_prefix(f"{metric_str}_")

prec_flux_upper_df.to_csv(drive_path+"/timeseries_csv/prec_flux_uppershare.csv")
#%% for lower breaches of prec_flux, we need only the counts since the extent of breach says nothing about dryness

lowerbreaches = xr.open_mfdataset(f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/lowerbreaches*.nc", combine="by_coords")
thresh_type="lower"
lowerbreaches = lowerbreaches["lower_breach"].where(take_region)
lowerbreaches=lowerbreaches.to_dataset()
crop="arabica"
prec_flux_lower_dict=make_climate_compressed_timeseries(crop=crop, climate_df=lowerbreaches)
dfs_list=[]
for item in prec_flux_lower_dict:
    dfs_list.append(prec_flux_lower_dict[item])
prec_flux_lower_df=pd.concat(dfs_list,axis=0)
prec_flux_lower_df

prec_flux_lower_df.to_csv(drive_path+"/timeseries_csv/prec_flux_lowershare.csv")

prec_flux_lower_df=pd.read_csv(drive_path+"/timeseries_csv/prec_flux_lowershare.csv")
prec_flux_lower_df=prec_flux_lower_df.set_index("Unnamed: 0")
prec_flux_lower_df=prec_flux_lower_df.add_prefix(f"{metric_str}_")
prec_flux_lower_df.to_csv(drive_path+"/timeseries_csv/prec_flux_lowershare.csv")

#%% upper breaches for VPD
metric_str="vpd_max"
zarr_path=f"{drive_path}/{metric_str}/brazil/vpd_processed_batches/*.zarr"

main_call=zarr_grid_descriptives(metric_str=metric_str, area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)
main_zarr=main_call.main_zarr
take_region=main_zarr.notnull().any("time")
take_region=take_region.load() 
#take_region["prec_flux"].plot()
take_region = take_region[metric_str]    

#%% vpd_max extent of breach


thresh_type='upper'
threshold_path=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_*.nc"
upperbreaches = xr.open_mfdataset(f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/upperbreach*mag*.nc", combine="by_coords")
upperthresh = xr.open_mfdataset(threshold_path)

upperbreaches = upperbreaches["upper_breach_mag"].where(take_region)
upperthresh = upperthresh["upper"].where(take_region)

upperthresh = xr.where(upperthresh == 0, 0.01, upperthresh)   
upperextent = (upperbreaches / upperthresh).rename("upper_extent")
upperextent=upperextent.to_dataset()

crop="arabica"
vpd_max_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=upperextent)
dfs_list=[]
for item in vpd_max_upper_dict:
    dfs_list.append(vpd_max_upper_dict[item])
vpd_max_upper_df=pd.concat(dfs_list,axis=0)
vpd_max_upper_df=vpd_max_upper_df.add_prefix(metric_str)
vpd_max_upper_df.to_csv(drive_path+f"/timeseries_csv/{metric_str}_upperextent.csv")

#%% vpd_max breach share in quadrants over grids
metric_str="vpd_max"
upperbreaches = xr.open_mfdataset(f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/upperbreaches*.nc", combine="by_coords")
thresh_type="upper"
upperbreaches = upperbreaches["upper_breach"].where(take_region)
upperbreaches=upperbreaches.to_dataset()
crop="arabica"
vpd_max_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=upperbreaches)
dfs_list=[]
for item in vpd_max_upper_dict:
    dfs_list.append(vpd_max_upper_dict[item])
vpd_max_upper_df=pd.concat(dfs_list,axis=0)
vpd_max_upper_df=vpd_max_upper_df.add_prefix(metric_str)
vpd_max_upper_df.to_csv(drive_path+f"/timeseries_csv/{metric_str}_uppershare.csv")

#%% the daily gridded averages of the vpd and prec_flux
crop="arabica"
metric_str="vpd_max"
zarr_path=f"{drive_path}/{metric_str}/brazil/vpd_processed_batches/*.zarr"

main_call=zarr_grid_descriptives(metric_str=metric_str, area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)
vpd_daily=main_call.main_zarr
vpd_daily=vpd_daily.drop_vars("crs")

vpd_daily_dict=make_climate_compressed_timeseries(crop=crop, climate_df=vpd_daily)
dfs_list=[]
for item in vpd_daily_dict:
    dfs_list.append(vpd_daily_dict[item])
vpd_daily_df=pd.concat(dfs_list,axis=0)
vpd_daily_df=vpd_daily_df.add_prefix(metric_str)
vpd_daily_df.to_csv(drive_path+f"/timeseries_csv/{metric_str}_daily_mean_values.csv")

# =============================

metric_str="prec_flux"
zarr_path=f"{drive_path}/{metric_str}/brazil/prec_zarr_processed/*.zarr"

main_call=zarr_grid_descriptives(metric_str=metric_str, area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)
prec_flux_daily=main_call.main_zarr
prec_flux_daily=prec_flux_daily.drop_vars("crs")

prec_flux_daily_dict=make_climate_compressed_timeseries(crop=crop, climate_df=prec_flux_daily)
dfs_list=[]
for item in prec_flux_daily_dict:
    dfs_list.append(prec_flux_daily_dict[item])
prec_flux_daily_df=pd.concat(dfs_list,axis=0)
prec_flux_daily_df=prec_flux_daily_df.add_prefix(metric_str)
prec_flux_daily_df.to_csv(drive_path+f"/timeseries_csv/{metric_str}_daily_mean_values.csv")


#%% selecting the replacement for the 0 values of upper thresholds
times = pd.to_datetime(upperthresh.time.values)
n = upperthresh.sizes["time"]
batch_size = 1000
upperthresh = upperthresh["prec_flux"].where(take_region)     
upperbreaches = upperbreaches["prec_flux"].where(take_region)
upperthresh=upperthresh.drop_vars("crs")
upperbreaches=upperbreaches.drop_vars("crs")
for start_idx in range(0, n, batch_size):
    end_idx = min(start_idx + batch_size, n)                 # never past the end
    chunk = upperthresh.isel(time=slice(start_idx, end_idx))

    mn, mx, mn_pos = (chunk.min().compute(),
                      chunk.max().compute(),
                      chunk.where(chunk > 0).min().compute())
    mn,mx,mn_pos=mn["prec_flux"].item(),mx["prec_flux"].item(),mn_pos["prec_flux"].item()
    print(f"{times[start_idx]:%Y%m%d} to {times[end_idx - 1]:%Y%m%d}: "
          f"min={mn:.3g}, min>0={mn_pos:.3g}, max={mx:.3g}")
    """
    20000101 to 20020926: min=0, min>0=0.01, max=216
    20020927 to 20050622: min=0, min>0=0.01, max=172
    20050623 to 20080318: min=0, min>0=0.01, max=172
    20080319 to 20101213: min=0, min>0=0.01, max=157
    20101214 to 20130908: min=0, min>0=0.01, max=222
    20130909 to 20160604: min=0, min>0=0.01, max=170
    20160605 to 20190301: min=0, min>0=0.01, max=165
    20190302 to 20211125: min=0, min>0=0.01, max=222
    20211126 to 20240821: min=0, min>0=0.01, max=206
    20240822 to 20251231: min=0, min>0=0.01, max=160
    from here, we set the minimum upper threshold to be 0.01 wherever it is 0,
    to help normalise the breach
    """


num,den=list(upperbreaches.data_vars)[1],list(upperthresh.data_vars)[1]
print(num," ," ,den)
upperextent=upperbreaches[num]/upperthresh[den]
upperextent=upperextent.to_dataset()



