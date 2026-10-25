#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 16:44:57 2026

@author: darkmage
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:06:26 2026

@author: darkmage
"""

from climate_calculations import zarr_grid_descriptives, collect_net_cdf
import xarray as xr
import pandas as pd
hist_delta=5
hist_gap=5
margin_of_days=5

drive_path="/media/darkmage/SSD-Storage1/"
area_str="centralSA"
upper_quantile,lower_quantile=0.95,0.05
window="moving"

#%%
import zipfile, glob, os

files=sorted(glob.glob("/media/darkmage/SSD-Storage/prec_flux/prec_flux_raw/precipitation_flux_raw/*.zip"))
out = "/media/darkmage/SSD-Storage/prec_flux/main_prec_flux_raw"
for y in range(1990, 2000):                         # 1990 to 1999 inclusive
    for z in files:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        print("done:", z)
        
        
#%% 
"""
from netcdf_explore import assemble_zarr_batch

assemble=assemble_zarr_batch(netcdfs_path=f"{main_dir}/main_prec_flux_raw/*.nc",
                             output_path=f"{main_dir}/brazil/prec_zarr_processed/")
assemble.make_zarr_batches(area_str="centralSA",
                           metric_str="prec",
                           metric_rename="prec_flux")
"""
#%%
class the_thresholds:
    def __init__(self,t0,t1,thresh_type,desc_str,output_path,metric_str,
                 upper_quantile,lower_quantile,zarr_path,
                 area_str=area_str,
                 window=window,
                 hist_delta=hist_delta,hist_gap=hist_gap,
                 margin_of_days=margin_of_days,):
        self.t0,self.t1,self.metric_str,self.area_str=t0,t1,metric_str,area_str
        self.zarr_path=zarr_path
        self.upper_quantile=upper_quantile
        self.lower_quantile=lower_quantile
        self.window=window
        self.thresh_type=thresh_type
        self.hist_delta=hist_delta
        self.hist_gap=hist_gap
        self.margin_of_days=margin_of_days
        self.desc_str=desc_str
        self.output_path=output_path
        
    def the_thresholds(self):
        main_call=zarr_grid_descriptives(metric_str=self.metric_str, 
                                         area_str=self.area_str)
        main_call.get_daily_zarr_grids(zarr_path=self.zarr_path)
        
        if self.thresh_type=="upper":
            main_call.make_upper_moving_threshold(present_start_date=self.t0, 
                                                  present_end_date=self.t1,
                                                  hist_delta=self.hist_delta,
                                                  margin_of_days=self.margin_of_days,
                                                  hist_gap=self.hist_gap,
                                                  upper_quantile=self.upper_quantile)
            self.list_thresholds= main_call.moving_upper_thresholds
        elif self.thresh_type=="lower":
            main_call.make_lower_moving_threshold(present_start_date=self.t0, 
                                                  present_end_date=self.t1,
                                                  hist_delta=self.hist_delta,
                                                  margin_of_days=self.margin_of_days,
                                                  hist_gap=self.hist_gap,
                                                  lower_quantile=self.lower_quantile)
            self.list_thresholds= main_call.moving_lower_thresholds
        
        collect_net_cdf(list_of_gis=self.list_thresholds,
                        thresh_type=self.thresh_type,
                        lats=main_call.main_zarr.lat, 
                        lons=main_call.main_zarr.lon, 
                        metric_str=self.metric_str, 
                        area_str=self.area_str, 
                        description_str=self.desc_str, 
                        hist_delta=self.hist_delta, 
                        hist_gap=self.hist_gap, 
                        margin_of_days=self.margin_of_days, 
                        window=self.window, output_path=self.output_path)
#%% PREC FLUX DETAILS
metric_str="prec_flux"
zarr_path=f"{drive_path}/{metric_str}/brazil/prec_zarr_processed/*.zarr"
main_dir = "/media/darkmage/SSD-Storage1/prec_flux/"
zarr_path = f"{main_dir}/brazil/prec_zarr_processed/*.zarr"
#%% prec_flux lower 2021-01-1 to 2025-12-31

t0,t1="2021-01-01","2025-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="lower"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux lower thresholds from {t0} to {t1}"

main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()

#%% prec_flux upper 2021-01-1 to 2025-12-31

t0,t1="2021-01-01","2025-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()

#%% prec_flux lower 2016-01-01 to 2020-12-31

t0,t1="2016-01-01","2020-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="lower"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux upper 2016-01-01 to 2020-12-31

t0,t1="2016-01-01","2020-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux lower 2011-01-01 to 2015-12-31

t0,t1="2011-01-01","2015-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="lower"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux upper 2011-01-01 to 2015-12-31

t0,t1="2011-01-01","2015-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux lwoer 2006-01-01 to 2010-12-31

t0,t1="2006-01-01","2010-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="lower"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux upper 2006-01-01 to 2010-12-31

t0,t1="2006-01-01","2010-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux lower 2000-01-01 to 2005-12-31


t0,t1="2000-01-01","2005-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="lower"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% prec_flux upper 2000-01-01 to 2005-12-31

t0,t1="2000-01-01","2005-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{main_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% upper breach of prec_flux
main_call=zarr_grid_descriptives(metric_str="prec_flux", area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)

thresh_type="upper"
threshold_path=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_*.nc"
t0,t1="2000-01-01","2025-12-31"
main_call.process_and_save_breach_moving_batches(present_start_date=t0, present_end_date=t1, 
                                                 moving_thresholds_dir=threshold_path,
                                                 thresh_type=thresh_type,
                                                 output_dir=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/")
#%% lower breach of prec_flux
main_call=zarr_grid_descriptives(metric_str="prec_flux", area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)

thresh_type="lower"
threshold_path=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_*.nc"
t0,t1="2000-01-01","2025-12-31"
main_call.process_and_save_breach_moving_batches(present_start_date=t0, present_end_date=t1, 
                                                 moving_thresholds_dir=threshold_path,
                                                 thresh_type=thresh_type,
                                                 output_dir=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/")

#%% for VPD we only do the upper breaches since low VPD values are just normal and do not indicate stress points

vpd_dir = "/media/darkmage/SSD-Storage1/vpd_max/"
metric_str="vpd_max"
zarr_path=f"{drive_path}/{metric_str}/brazil/vpd_processed_batches/*.zarr"

#%% vpd upper 2021-01-01 to 2025-12-31
t0,t1="2021-01-01","2025-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{vpd_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,zarr_path=zarr_path,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()

#%% vpd upper 2016-01-01 to 2020-12-31
t0,t1="2016-01-01","2020-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{vpd_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,zarr_path=zarr_path,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% vpd upper 2011-01-01 to 2015-12-31

t0,t1="2011-01-01","2015-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{vpd_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,zarr_path=zarr_path,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()
#%% vpd upper 2006-01-01 to 2010-12-31

t0,t1="2006-01-01","2010-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{vpd_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,zarr_path=zarr_path,
                         upper_quantile=upper_quantile,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()

#%% vpd upper 2000-01-01 to 2005-12-31

t0,t1="2000-01-01","2005-12-31"
y0,y1=t0[2:4],t1[2:4]
thresh_type="upper"
output_path=f"{vpd_dir}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_thresholds_{y0}to{y1}.nc"
desc_str=f"Moving daily precipitation flux {thresh_type} thresholds from {t0} to {t1}"
main_call=the_thresholds(t0=t0, t1=t1,thresh_type=thresh_type,desc_str=desc_str,
                         output_path=output_path,metric_str=metric_str,
                         upper_quantile=upper_quantile,zarr_path=zarr_path,
                         lower_quantile=lower_quantile)
main_call.the_thresholds()

#%% upper breach of vpd_max
main_call=zarr_grid_descriptives(metric_str=metric_str, area_str="centralSA")
main_call.get_daily_zarr_grids(zarr_path=zarr_path)

thresh_type="upper"
threshold_path=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/moving_{thresh_type}_*.nc"
t0,t1="2000-01-01","2025-12-31"
main_call.process_and_save_breach_moving_batches(present_start_date=t0, present_end_date=t1, 
                                                 moving_thresholds_dir=threshold_path,
                                                 thresh_type=thresh_type,
                                                 output_dir=f"{drive_path}/{metric_str}/brazil/moving_threshold_window{margin_of_days}_delta{hist_delta}/breaches_excess/")