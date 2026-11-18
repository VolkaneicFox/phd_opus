

#%%
drive_path=""
hist_delta=20
margin_of_days=5
import xarray as xr 
import numpy as  np
import cupy as cp
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
from climate_calculations import zarr_grid_descriptives

"""
will drop 29 Feb Calendar Dates
"""
#%% calculating the vpd_max upper breach

metric_str = "vpd_max"
thresh_type="upper"

present_years = range(2005,2026)
for slice_no in [1,2,3,4]:

    thresholds = drive_path+metric_str+f"/brazil/moving_threshold_window5_delta20/slice{slice_no}_moving_upper_thresholds_05to25.nc"
    x=xr.open_dataset(thresholds)
    filenames=[drive_path+metric_str+f"/brazil/vpd_processed_batches/slice{slice_no}_yearly_vpd_max_brazil_{year}.zarr" for year in range(2005,2026)]
    yearfile = xr.open_mfdataset(filenames)
    output_path = drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"
    main_call = zarr_grid_descriptives(metric_str=metric_str, area_str=f"centralSA.{slice_no}")
    main_call.process_and_save_breach_moving_batches(now_zarr=yearfile[metric_str], 
                                                     thresh_zarr=x[thresh_type], 
                                                     thresh_type=thresh_type, 
                                                     slice_no=slice_no, 
                                                     output_dir=output_path)
    

#%% calculating the prec_flux upper and lower breaches

metric_str = "prec_flux"
present_years = range(2005,2026)
for slice_no in [1,2,3,4]:
    for thresh_type in ["upper","lower"]:

        thresholds = drive_path+metric_str+f"/brazil/moving_threshold_window5_delta20/slice{slice_no}_moving_{thresh_type}_thresholds_05to25.nc"
        x=xr.open_dataset(thresholds)
        filenames=[drive_path+metric_str+f"/brazil/prec_zarr_processed/slice{slice_no}_yearly_{metric_str}_brazil_{year}.zarr" for year in range(2005,2026)]
        yearfile = xr.open_mfdataset(filenames)
        output_path = drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"
        main_call = zarr_grid_descriptives(metric_str=metric_str, area_str=f"centralSA.{slice_no}")
        main_call.process_and_save_breach_moving_batches(now_zarr=yearfile[metric_str], 
                                                         thresh_zarr=x[thresh_type], 
                                                         thresh_type=thresh_type, 
                                                         slice_no=slice_no, 
                                                         output_dir=output_path)
        
#%%
snap = lambda da: da.assign_coords(lat=da.lat.round(1), lon=da.lon.round(1))
from grid_compression import weighted_time_series
def make_climate_compressed_timeseries(crop,climate_df,metric_str):
    phase_periods={"2005to09":("2005-01-01","2009-12-31"),
                  "2010to20":("2010-01-01","2019-12-31"),
                  "2020to25":("2020-01-01","2025-12-31")}
    
    df_dict={}
    grids_weight_path=drive_path+"mapspam_grids"
    
    for phase in phase_periods:
        t0,t1=phase_periods[phase][0],phase_periods[phase][1]
    
        time_series=weighted_time_series(crop=crop, 
                                         period=phase, metric_str=metric_str,
                                         grids_weights_path=grids_weight_path)
        climate_zarr=climate_df.sel(time=slice(t0,t1))
        time_series.align_the_dimensions(climate_zarr=climate_zarr)
        df_dict[phase]=time_series.save_the_time_series()
    return df_dict
#%%  time series being compressed into crop weighted quadrants

"""
For prec-flux, the lower breach is binary. For the VPD and Prec-Flux, the upper 
breach is considered in binary and in magnitude, to take the percentage breach, 
in prec-flux, set the minimum values of upper-threshold, where it is 0 to be 0.01
"""

metric_str = "prec_flux"


prec_flux_up=[drive_path+metric_str+f"/brazil/moving_threshold_window5_delta20/slice{i}_moving_upper_thresholds_05to25.nc" for i in [1,2,3,4]]
prec_flux_up = xr.open_mfdataset(prec_flux_up)
prec_flux_up = xr.where(prec_flux_up==0, 0.01, prec_flux_up)

prec_flux_up_breach=os.listdir(drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/")
prec_flux_up_breach=[drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"+i for i in prec_flux_up_breach if "upperbreachmag_" in i]
prec_flux_up_breach=xr.open_mfdataset(prec_flux_up_breach)

prec_flux_up_breach, prec_flux_up = snap(prec_flux_up_breach), snap(prec_flux_up)
num = prec_flux_up_breach["upper_breach_mag"]
den = prec_flux_up["upper"]
res_prec_flux = num / den.where(den != 0)
res_prec_flux.name = "upper_breach_extent_prec_flux"
res_prec_flux=res_prec_flux.to_dataset()

prec_flux_up_is=os.listdir(drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/")
prec_flux_up_is=[drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"+i for i in prec_flux_up_is if "upperbreaches_" in i]
prec_flux_up_is = xr.open_mfdataset(prec_flux_up_is)

prec_flux_low_is=os.listdir(drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/")
prec_flux_low_is=[drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"+i for i in prec_flux_low_is if "lowerbreaches_" in i]
prec_flux_low_is = xr.open_mfdataset(prec_flux_low_is)

plot_date = "2025-06-12"

prec_flux_up["upper"].sel(time=plot_date).plot() ## not nan in oceans, needed for the weighting
prec_flux_up_breach["upper_breach_mag"].sel(time=plot_date).plot() ## not nan in oceans, needed for the weighting

crop="arabica"
prec_flux_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=res_prec_flux,
                                                        metric_str=metric_str)
dfs_list=[]
for item in prec_flux_upper_dict:
    dfs_list.append(prec_flux_upper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df.columns = ['arup_pr_area1', 'arup_pr_area2', 'arup_pr_area3', 'arup_pr_area4'] ## metric string was vpd_max
prec_flux_upper_df.to_csv(drive_path+"/timeseries_csv/prec_flux_upperextent_w5d20.csv")

prec_flux_exupper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=prec_flux_up_is,
                                                        metric_str=metric_str)
dfs_list=[]
for item in prec_flux_exupper_dict:
    dfs_list.append(prec_flux_exupper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df.to_csv(drive_path+"/timeseries_csv/prec_flux_uppershare_w5d20.csv")

## prec flux 0 days

metric_str = "prec_flux"
present_years = range(2005,2026)
filenames=[drive_path+metric_str+f"/brazil/prec_zarr_processed/slice{slice_no}_yearly_{metric_str}_brazil_{year}.zarr" for year in range(2005,2026) for slice_no in [1,2,3,4]]
yearfile = xr.open_mfdataset(filenames)
pfis0 = (yearfile[metric_str]==0).astype(int)
pfis0 = pfis0.to_dataset()      

pfis0_dict=make_climate_compressed_timeseries(crop=crop, climate_df=pfis0,
                                                        metric_str=metric_str)
dfs_list=[]
for item in pfis0_dict:
    dfs_list.append(pfis0_dict[item])
pfis0_df=pd.concat(dfs_list,axis=0)
pfis0_df.to_csv(drive_path+"/timeseries_csv/pfis0.csv")

crop="soybean"
prec_flux_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=res_prec_flux,
                                                        metric_str=metric_str)
dfs_list=[]
for item in prec_flux_upper_dict:
    dfs_list.append(prec_flux_upper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df.columns = ['soup_pr_area1', 'soup_pr_area2', 'soup_pr_area3', 'soup_pr_area4'] ## metric string was vpd_max
prec_flux_upper_df.to_csv(drive_path+f"/timeseries_csv/{crop}prec_flux_upperextent_w5d20.csv")

prec_flux_exupper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=prec_flux_up_is,
                                                        metric_str=metric_str)
dfs_list=[]
for item in prec_flux_exupper_dict:
    dfs_list.append(prec_flux_exupper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df.to_csv(drive_path+f"/timeseries_csv/{crop}prec_flux_uppershare_w5d20.csv")

## prec flux 0 days

metric_str = "prec_flux"
present_years = range(2005,2026)
filenames=[drive_path+metric_str+f"/brazil/prec_zarr_processed/slice{slice_no}_yearly_{metric_str}_brazil_{year}.zarr" for year in range(2005,2026) for slice_no in [1,2,3,4]]
yearfile = xr.open_mfdataset(filenames)
pfis0 = (yearfile[metric_str]==0).astype(int)
pfis0 = pfis0.to_dataset()      

pfis0_dict=make_climate_compressed_timeseries(crop=crop, climate_df=pfis0,
                                                        metric_str=metric_str)
dfs_list=[]
for item in pfis0_dict:
    dfs_list.append(pfis0_dict[item])
pfis0_df=pd.concat(dfs_list,axis=0)
pfis0_df.to_csv(drive_path+f"/timeseries_csv/{crop}pfis0.csv")

prec_flux_low_dict=make_climate_compressed_timeseries(crop=crop, climate_df=prec_flux_low_is,
                                                        metric_str=metric_str)
dfs_list=[]
for item in prec_flux_low_dict:
    dfs_list.append(prec_flux_low_dict[item])
prec_flux_low_df=pd.concat(dfs_list,axis=0)
prec_flux_low_df.to_csv(drive_path+f"/timeseries_csv/{crop}prec_flux_lowshare_w5d20.csv")


"=============================="
metric_str = "vpd_max"

vpd_max_up=[drive_path+metric_str+f"/brazil/moving_threshold_window5_delta20/slice{i}_moving_upper_thresholds_05to25.nc" for i in [1,2,3,4]]
vpd_max_up = xr.open_mfdataset(vpd_max_up)
vpd_max_up = xr.where(vpd_max_up==0, 0.01, vpd_max_up)

vpd_max_up_breach=os.listdir(drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/")
vpd_max_up_breach=[drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"+i for i in vpd_max_up_breach if "upperbreachmag_" in i]
vpd_max_up_breach=xr.open_mfdataset(vpd_max_up_breach)

vpd_max_up_breach, vpd_max_up = snap(vpd_max_up_breach), snap(vpd_max_up)
num = vpd_max_up_breach["upper_breach_mag"]
den = vpd_max_up["upper"]
res_vpd_max = num / den.where(den != 0)
res_vpd_max.name = "upper_breach_extent_vpd_max"
res_vpd_max=res_vpd_max.to_dataset()

vpd_max_up_is=os.listdir(drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/")
vpd_max_up_is=[drive_path+metric_str+"/brazil/moving_threshold_window5_delta20/breach_excess/"+i for i in vpd_max_up_is if "upperbreaches_" in i]
vpd_max_up_is = xr.open_mfdataset(vpd_max_up_is)

crop="arabica"
vpd_max_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=vpd_max_up_is,
                                                        metric_str=metric_str)

dfs_list=[]
for item in vpd_max_upper_dict:
    dfs_list.append(vpd_max_upper_dict[item])
vpd_max_upper_df=pd.concat(dfs_list,axis=0)
vpd_max_upper_df.to_csv(drive_path+f"/timeseries_csv/{crop}vpd_max_uppershare_w5d20.csv")

vpd_max_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=res_vpd_max,
                                                        metric_str=metric_str)
dfs_list=[]
for item in vpd_max_upper_dict:
    dfs_list.append(vpd_max_upper_dict[item])
vpd_max_upper_df=pd.concat(dfs_list,axis=0)
vpd_max_upper_df.to_csv(drive_path+f"/timeseries_csv/{crop}vpd_max_upperextent_w5d20.csv")

crop="soybean"
vpd_max_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=vpd_max_up_is,
                                                        metric_str=metric_str)

dfs_list=[]
for item in vpd_max_upper_dict:
    dfs_list.append(vpd_max_upper_dict[item])
vpd_max_upper_df=pd.concat(dfs_list,axis=0)
vpd_max_upper_df.to_csv(drive_path+f"/timeseries_csv/{crop}vpd_max_uppershare_w5d20.csv")

vpd_max_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=res_vpd_max,
                                                        metric_str=metric_str)
dfs_list=[]
for item in vpd_max_upper_dict:
    dfs_list.append(vpd_max_upper_dict[item])
vpd_max_upper_df=pd.concat(dfs_list,axis=0)
vpd_max_upper_df.to_csv(drive_path+f"/timeseries_csv/{crop}vpd_max_upperextent_w5d20.csv")


#%% comparison between the regional share (binary) upepr threshold prec_flux breach, with hist_delta =5 vs 20
prec_uppershared20=pd.read_csv(drive_path+"/timeseries_csv/prec_flux_uppershare_w5d20.csv")
prec_uppershared20["Unnamed: 0"]=pd.to_datetime(prec_uppershared20["Unnamed: 0"])
prec_uppershared20=prec_uppershared20.set_index("Unnamed: 0")
prec_uppershared20=prec_uppershared20.rename(columns={ 'arup_pr_area1':"Pfd20UPsh_ar1", 
                                                       'arup_pr_area2':"Pfd20UPsh_ar2", 
                                                       'arup_pr_area3':"Pfd20UPsh_ar3",
                                                       'arup_pr_area4':"Pfd20UPsh_ar4"})

prec_uppershared5=pd.read_csv(drive_path+"/timeseries_csv/prec_flux_uppershare.csv")
prec_uppershared5["Unnamed: 0"]=pd.to_datetime(prec_uppershared5["Unnamed: 0"])
prec_uppershared5=prec_uppershared5.set_index("Unnamed: 0")
prec_uppershared5=prec_uppershared5.rename(columns={'prec_flux_ara_upp_area1':"Pfd5UPsh_ar1", 
                                                    'prec_flux_ara_upp_area2':"Pfd5UPsh_ar2", 
                                                    'prec_flux_ara_upp_area3':"Pfd5UPsh_ar3",
                                                    'prec_flux_ara_upp_area4':"Pfd5UPsh_ar4"})
# 3. Plot a side-by-side or overlaid comparison for Area 1 (or any area you like)
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

# Plot Area 1 comparison as an example
axes[0].plot(prec_uppershared20.index, prec_uppershared20["Pfd20UPsh_ar1"], label="20-Year Baseline Window", color="#1f77b4", alpha=0.8, linewidth=1.5)
axes[0].plot(prec_uppershared5.index, prec_uppershared5["Pfd5UPsh_ar1"], label="5-Year Baseline Window", color="#ff7f0e", alpha=0.8, linewidth=1.5)
axes[0].set_title("Upper Breach Share Comparison: Area 1 (20-Yr vs. 5-Yr Baseline)")
axes[0].set_ylabel("Breach Share")
axes[0].legend()
axes[0].grid(True, linestyle="--", alpha=0.5)

# Plot all 4 areas together for the 20-year window to check spatial variance
for col in prec_uppershared20.columns:
    axes[1].plot(prec_uppershared20.index, prec_uppershared20[col], label=col, alpha=0.8, linewidth=1.2)
axes[1].set_title("20-Year Baseline Window Across All 4 Quadrant Areas")
axes[1].set_ylabel("Breach Share")
axes[1].set_xlabel("Time")
axes[1].legend(loc="upper left")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()



prec_flux_upper_dict=make_climate_compressed_timeseries(crop=crop, climate_df=res_prec_flux,
                                                        metric_str=metric_str)
dfs_list=[]
for item in prec_flux_upper_dict:
    dfs_list.append(prec_flux_upper_dict[item])
prec_flux_upper_df=pd.concat(dfs_list,axis=0)
prec_flux_upper_df.to_csv(drive_path+"/timeseries_csv/prec_flux_upperextent_w5d20.csv")




#%% Precipitation Flux
metric_str="prec_flux"
zarr_path=f"{drive_path}/{metric_str}/brazil/prec_zarr_processed/"
take_region = xr.open_dataset(zarr_path+"yearly_prec_flux_brazil_2022.zarr")
take_region=take_region.notnull().any("time")
take_region=take_region.load() 
take_region["prec_flux"].plot()
take_region = take_region[metric_str]    
#%% prec flux extent of breach over grids in quadrants
thresh_type='upper'
threshold_path=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_*.nc"
upperbreaches = xr.open_mfdataset(f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/upperbreach*mag*.nc", 
                                  combine="by_coords")
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

#%% the daily gridded averages of the vpd and prec_flux
crop="soybean"
metric_str="vpd_max"
present_years = range(2005,2026)
filenames=[drive_path+metric_str+f"/brazil/vpd_processed_batches/slice{slice_no}_yearly_{metric_str}_brazil_{year}.zarr" for year in range(2005,2026) for slice_no in [1,2,3,4]]
vpd_daily = xr.open_mfdataset(filenames)

vpd_daily_dict=make_climate_compressed_timeseries(crop=crop, climate_df=vpd_daily, metric_str=metric_str)
dfs_list=[]
for item in vpd_daily_dict:
    dfs_list.append(vpd_daily_dict[item])
vpd_daily_df=pd.concat(dfs_list,axis=0)
vpd_daily_df=vpd_daily_df.add_prefix(metric_str)
vpd_daily_df.to_csv(drive_path+f"/timeseries_csv/{crop}{metric_str}_daily_mean_values.csv")

# =============================

metric_str="prec_flux"
present_years = range(2005,2026)
filenames=[drive_path+metric_str+f"/brazil/prec_zarr_processed/slice{slice_no}_yearly_{metric_str}_brazil_{year}.zarr" for year in range(2005,2026) for slice_no in [1,2,3,4]]
prec_flux_daily = xr.open_mfdataset(filenames)

prec_flux_daily_dict=make_climate_compressed_timeseries(crop=crop, climate_df=prec_flux_daily, metric_str=metric_str
                                                        )
dfs_list=[]
for item in prec_flux_daily_dict:
    dfs_list.append(prec_flux_daily_dict[item])
prec_flux_daily_df=pd.concat(dfs_list,axis=0)
prec_flux_daily_df=prec_flux_daily_df.add_prefix(metric_str)
prec_flux_daily_df.to_csv(drive_path+f"/timeseries_csv/{crop}{metric_str}_daily_mean_values.csv")


#%%extra: selecting the replacement for the 0 values of upper thresholds for prec_flux
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



#%%

