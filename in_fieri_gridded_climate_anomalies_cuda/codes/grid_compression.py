#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 09:00:31 2026

@author: darkmage

the relevant grids are: lat= [-5,-20] ,[-20,-35] and lon = [-75,-55],[-55,-35]
Physical area (A)
All technologies (A)
Physical  area is measured  in a hectare and represents the actual area where a 
crop   is  grown, not  counting  how  often production was   harvested from it. 
Physical area is calculated for each production system and crop, and the sum of 
all physical areas of the four production systems constitute the total physical 
area for  that crop. The sum of  the physical areas of all crops in a pixel may 
not be larger than the pixel size.
"""

#%%
"""
mapspam grids are in the times as follows:
    
step, make all the year to year weights for the crops and then save them in 
"/paper_three/crop_area_weights/<crop_name>_<year-to-year>_<area_x>.nc"
where the area indices are,
    x=1: (minx=-75, miny=-35, maxx=-55, maxy=-20)
    x=2: (minx=-55, miny=-35, maxx=-35, maxy=-20)
    x=3: (minx=-75, miny=-20, maxx=-55, maxy=-5)
    x+4: (minx=-55, miny=-20, maxx=-35, maxy=-5)
"""
import rioxarray as rx
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
"""
FULL DATASETS
"""

class the_growing_climate_areas:
    def __init__(self,climate_zarr_path=""):
        #self.climate_zarr = xr.open_mfdataset(climate_zarr_path)
        None
        
    def physical_growing_areas(self,spam_path,crop,period):
        self.full_region=rx.open_rasterio(spam_path,engine="raster_io",masked=True).squeeze("band",drop=True) ## given growing regions
        #self.full_region=self.full_region.rio.clip_box(minx=-75, miny=-35, maxx=-35, maxy=-5)
        self.full_region.plot()
        title_str=f"{crop}: {period}"
        plt.title(title_str)
        plt.show()
        #self.climate_zarr=self.climate_zarr.sel(time=slice(start_time,end_time))
        self.crop=crop
        self.period=period

    def save_weight_figures(self,save_path):
        i=1
        title_str=f"{self.crop}: {self.period}"
        for w in [self.weights_1,
                  self.weights_2,
                  self.weights_3,
                  self.weights_4]:
            fig,ax=plt.subplots(figsize=(10,10))
            w.plot()
            plt.suptitle(f"{title_str}_area_{i}")
            plt.show()
            fig.savefig(f"{save_path}/{self.crop}_{self.period}_area_{i}.png",dpi=150)
            plt.close(fig)

    def partition_and_weights(self):        
        self.slice_1 = self.full_region.rio.clip_box(minx=-75, miny=-35, maxx=-55, maxy=-20).astype("float64")
        self.slice_2 = self.full_region.rio.clip_box(minx=-55, miny=-35, maxx=-35, maxy=-20).astype("float64")
        self.slice_3 = self.full_region.rio.clip_box(minx=-75, miny=-20, maxx=-55, maxy=-5).astype("float64")
        self.slice_4 = self.full_region.rio.clip_box(minx=-55, miny=-20, maxx=-35, maxy=-5).astype("float64")
        
        self.weights_1 = self.slice_1/self.slice_1.sum()
        self.weights_2 = self.slice_2/self.slice_2.sum()
        self.weights_3 = self.slice_3/self.slice_3.sum()
        self.weights_4 = self.slice_4/self.slice_4.sum()
        
        assert np.isclose(self.weights_1.sum().item(), 1.0)
        assert np.isclose(self.weights_2.sum().item(), 1.0)
        assert np.isclose(self.weights_3.sum().item(), 1.0)
        assert np.isclose(self.weights_4.sum().item(), 1.0)

    def _compress_grids(self,w):
        w["y"]=np.round(w["y"],1)
        w["x"]=np.round(w["x"],1)
        w_meaned=w.groupby(["y","x"]).mean()
        w_meaned=w_meaned.rename({"x": "lon", "y": "lat"})
        return w_meaned
    
    def compress_grids(self,grids_weights_path,if_save=True):    
        self.w_meaned_1=self._compress_grids(self.weights_1)
        self.w_meaned_2=self._compress_grids(self.weights_2)
        self.w_meaned_3=self._compress_grids(self.weights_3)
        self.w_meaned_4=self._compress_grids(self.weights_4)
        if if_save:
            self.w_meaned_1.to_netcdf(f"{grids_weights_path}/{self.crop}_{self.period}_area_1.nc")
            self.w_meaned_2.to_netcdf(f"{grids_weights_path}/{self.crop}_{self.period}_area_2.nc")
            self.w_meaned_3.to_netcdf(f"{grids_weights_path}/{self.crop}_{self.period}_area_3.nc")
            self.w_meaned_4.to_netcdf(f"{grids_weights_path}/{self.crop}_{self.period}_area_4.nc")
        else:
            print("not saving")

class weighted_time_series:
    
    def __init__(self,crop,period,grids_weights_path):
        self.crop=crop
        self.period=period
        self.w_meaned_1=xr.open_dataset(f"{grids_weights_path}/{self.crop}_{self.period}_area_1.nc")
        self.w_meaned_2=xr.open_dataset(f"{grids_weights_path}/{self.crop}_{self.period}_area_2.nc")
        self.w_meaned_3=xr.open_dataset(f"{grids_weights_path}/{self.crop}_{self.period}_area_3.nc")
        self.w_meaned_4=xr.open_dataset(f"{grids_weights_path}/{self.crop}_{self.period}_area_4.nc")
        
    def _align_the_dimensions(self, wt, cl):
        wt = wt.assign_coords(lat=wt.lat.round(2), lon=wt.lon.round(2))
        cl = cl.assign_coords(lat=cl.lat.round(2), lon=cl.lon.round(2))
    
        wt = wt.fillna(0)
        w_a = wt.reindex(lat=cl.lat, lon=cl.lon, fill_value=0)
    
        #kept = np.round(w_a.sum() / wt.sum(),4)
        #print(f"weight kept: {kept} | weight lost: {1 - kept}")
    
        self.w_a = w_a / w_a.sum()
        self.w_a = self.w_a.to_dataarray()
        return cl[self.metric_str].weighted(self.w_a).mean(["lat", "lon"])
    
    def align_the_dimensions(self,climate_zarr):
        self.climate_zarr=climate_zarr
        self.metric_str=list(self.climate_zarr.data_vars)[0]
        self.climate_1 = self.climate_zarr.sel(lat=slice(-20,-35),lon=slice(-75,-55))
        self.climate_2 = self.climate_zarr.sel(lat=slice(-20,-35),lon=slice(-55,-35))
        self.climate_3 = self.climate_zarr.sel(lat=slice(-5,-20),lon=slice(-75,-55))
        self.climate_4 = self.climate_zarr.sel(lat=slice(-5,-20),lon=slice(-55,-35))
        
        self.w1=self._align_the_dimensions(wt=self.w_meaned_1,cl=self.climate_1)
        self.w2=self._align_the_dimensions(wt=self.w_meaned_2,cl=self.climate_2)
        self.w3=self._align_the_dimensions(wt=self.w_meaned_3,cl=self.climate_3)
        self.w4=self._align_the_dimensions(wt=self.w_meaned_4,cl=self.climate_4)
        
    
    def save_the_time_series(self):
        crop_suffix=self.crop[:3]
        metric_suffix=self.metric_str[:3]
        self.dataframe=pd.DataFrame(index=self.w1.time.values,
                               columns=[f"{crop_suffix}_{metric_suffix}_area1",
                                        f"{crop_suffix}_{metric_suffix}_area2",
                                        f"{crop_suffix}_{metric_suffix}_area3",
                                        f"{crop_suffix}_{metric_suffix}_area4"])
        self.dataframe[f"{crop_suffix}_{metric_suffix}_area1"]=self.w1.values
        self.dataframe[f"{crop_suffix}_{metric_suffix}_area2"]=self.w2.values
        self.dataframe[f"{crop_suffix}_{metric_suffix}_area3"]=self.w3.values
        self.dataframe[f"{crop_suffix}_{metric_suffix}_area4"]=self.w4.values
        
        return self.dataframe
        
        

# %% COFFEE WEIGHTED GRIDS
def making_arabica_weights():
    crop="arabica"
    spam_path=f"/media/darkmage/SSD-Storage1/spam_raw_geotiffs/all_{crop}_grids/"
    """
    import os
    os.listdir(spam_path)
    """
    main_grow=the_growing_climate_areas()
    arabica_grids={'spam2020_V2r2_global_A_COFF_A.tif':"2020to25",
                   'spam2010V2r0_global_A_ACOF_A.tif':"2010to20",
                   'SPAM2005V3r2_global_A_TA_ACOF_A.tif':"2005to09"}
    for grid in arabica_grids:
        main_grow.physical_growing_areas(spam_path=spam_path+grid, 
                                     crop=crop, period=arabica_grids[grid])
        main_grow.partition_and_weights()
        main_grow.compress_grids(grids_weights_path="/media/darkmage/SSD-Storage1/mapspam_grids/",if_save=True)
        main_grow.save_weight_figures(save_path="/media/darkmage/SSD-Storage1/plots_and_figures")

