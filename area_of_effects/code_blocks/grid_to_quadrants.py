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
    
    def __init__(self,crop,period,metric_str,grids_weights_path):
        self.crop=crop
        self.period=period
        self.metric_str=metric_str
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
        return cl[self.thresh_type].weighted(self.w_a).mean(["lat", "lon"])
    
    def align_the_dimensions(self,climate_zarr):
        self.climate_zarr=climate_zarr
        self.thresh_type=list(self.climate_zarr.data_vars)[0] ## is taking up from upper
        self.climate_1 = self.climate_zarr.sel(lat=slice(-20,-35),lon=slice(-75,-55))
        self.climate_2 = self.climate_zarr.sel(lat=slice(-20,-35),lon=slice(-55,-35))
        self.climate_3 = self.climate_zarr.sel(lat=slice(-5,-20),lon=slice(-75,-55))
        self.climate_4 = self.climate_zarr.sel(lat=slice(-5,-20),lon=slice(-55,-35))
        
        self.w1=self._align_the_dimensions(wt=self.w_meaned_1,cl=self.climate_1)
        self.w2=self._align_the_dimensions(wt=self.w_meaned_2,cl=self.climate_2)
        self.w3=self._align_the_dimensions(wt=self.w_meaned_3,cl=self.climate_3)
        self.w4=self._align_the_dimensions(wt=self.w_meaned_4,cl=self.climate_4)
        
    
    def save_the_time_series(self):
        crop_suffix=self.crop[:2]
        metric_suffix=self.metric_str[:2]
        thresh_suffix=self.thresh_type[:2]
        self.dataframe=pd.DataFrame(index=self.w1.time.values,
                               columns=[f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area1",
                                        f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area2",
                                        f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area3",
                                        f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area4"])
        self.dataframe[f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area1"]=self.w1.values
        self.dataframe[f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area2"]=self.w2.values
        self.dataframe[f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area3"]=self.w3.values
        self.dataframe[f"{crop_suffix}{thresh_suffix}_{metric_suffix}_area4"]=self.w4.values
        
        return self.dataframe
        
drive_path=""      

# %% COFFEE WEIGHTED GRIDS
def making_arabica_weights():
    crop="arabica"
    spam_path=f"{drive_path}/spam_raw_geotiffs/all_{crop}_grids/"
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
        main_grow.compress_grids(grids_weights_path=f"{drive_path}/mapspam_grids/",if_save=True)
        main_grow.save_weight_figures(save_path=f"{drive_path}/plots_and_figures")
        
#%% SOYBEAN WEIGHTED GRIDS

def making_soybean_weights():
    crop="soybean"
    spam_path=f"{drive_path}/spam_raw_geotiffs/all_{crop}_grids/"
    main_grow=the_growing_climate_areas()
    soybean_grids={'spam2020_V2r2_global_A_SOYB_A.tif':"2020to25",
                   'spam2010V2r0_global_A_SOYB_A.tif':"2010to20",
                   'SPAM2005V3r2_global_A_TA_SOYB_A.tif':"2005to09"}
    for grid in soybean_grids:
        main_grow.physical_growing_areas(spam_path=spam_path+grid, crop=crop, period=soybean_grids[grid])
        main_grow.partition_and_weights()
        main_grow.compress_grids(grids_weights_path=f"{drive_path}/mapspam_grids/",if_save=True)
        
#%% PRINT THE SOYBEAN QUADRANT, PERIOD TOTAL AREA UNDER CULTIVATION
def print_the_quadrant_areas(grids, crop):
    """
    Computes quadrant areas across multiple periods (phases) and outputs
    a single unified publication-quality LaTeX comparison table.
    'grids' is expected to be a dictionary or ordered mapping, e.g., 
    { 'Phase 1': 'file_2005to09.tif', 'Phase 2': 'file_2010to20.tif', 'Phase 3': 'file_2020to25.tif' }
    """
    spam_path = f"{drive_path}/spam_raw_geotiffs/all_{crop}_grids/"
    
    quadrant_regions = {
        1: "Argentina (Córdoba, Santa Fe, Santiago del Estero), Paraguay (Boquerón), Bolivia (Tarija), Brazil (Mato Grosso do Sul, Paraná)",
        2: "Brazil (São Paulo, Paraná, Santa Catarina, Rio Grande do Sul), Uruguay, Argentina (Buenos Aires, Entre Ríos)",
        3: "Bolivia (Santa Cruz, Beni), Peru (Ucayali, Madre de Dios), Brazil (Rondônia, Mato Grosso), Paraguay (Concepción)",
        4: "Brazil (Mato Grosso, Goiás, Minas Gerais, Bahia, Tocantins, Maranhão, Piauí)"
    }

    # Store results per phase
    phase_data = {}
    phase_totals = {}

    for filename,phase_name in grids.items():
        full_region = rx.open_rasterio(spam_path + filename, engine="raster_io", masked=True).squeeze("band", drop=True)
        
        s1 = np.round(full_region.rio.clip_box(minx=-75, miny=-35, maxx=-55, maxy=-20).astype("float64").sum().item() / 100, 2)
        s2 = np.round(full_region.rio.clip_box(minx=-55, miny=-35, maxx=-35, maxy=-20).astype("float64").sum().item() / 100, 2)
        s3 = np.round(full_region.rio.clip_box(minx=-75, miny=-20, maxx=-55, maxy=-5).astype("float64").sum().item() / 100, 2)
        s4 = np.round(full_region.rio.clip_box(minx=-55, miny=-20, maxx=-35, maxy=-5).astype("float64").sum().item() / 100, 2)
        
        tot = np.round(s1 + s2 + s3 + s4, 2)
        phase_totals[phase_name] = tot
        
        phase_data[phase_name] = {
            1: (s1, f"{np.round((s1/tot)*100, 2):.2f}\\%" if tot > 0 else "0.00\\%"),
            2: (s2, f"{np.round((s2/tot)*100, 2):.2f}\\%" if tot > 0 else "0.00\\%"),
            3: (s3, f"{np.round((s3/tot)*100, 2):.2f}\\%" if tot > 0 else "0.00\\%"),
            4: (s4, f"{np.round((s4/tot)*100, 2):.2f}\\%" if tot > 0 else "0.00\\%")
        }

    phase_keys = list(grids.values()) # Expects 3 phases
    p1, p2, p3 = phase_keys[0], phase_keys[1], phase_keys[2]

    # Build LaTeX table string
    latex_table = f"""\\begin{{table*}}[htbp]
  \\centering
  \\caption{{Multi-phase spatial distribution and cultivation area of {crop.capitalize()} across quadrants}}
  \\label{{tab:multi_phase_quadrant_areas}}
  \\begin{{tabular}}{{l p{{5.0cm}} r r r r r r}}
    \\toprule
    \\textbf{{Quadrant}} & \\textbf{{Areas in Quadrant (Countries \\& Provinces)}} & \\multicolumn{{2}}{{c}}{{\\textbf{{{p1}}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{{p2}}}}} & \\multicolumn{{2}}{{c}}{{\\textbf{{{p3}}}}} \\\\
    \\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}} \\cmidrule(lr){{7-8}}
    & & \\textbf{{Area (km$^2$)}} & \\textbf{{Share}} & \\textbf{{Area (km$^2$)}} & \\textbf{{Share}} & \\textbf{{Area (km$^2$)}} & \\textbf{{Share}} \\\\
    \\midrule
    Quadrant 1 & {quadrant_regions[1]} & {phase_data[p1][1][0]:,.2f} & {phase_data[p1][1][1]} & {phase_data[p2][1][0]:,.2f} & {phase_data[p2][1][1]} & {phase_data[p3][1][0]:,.2f} & {phase_data[p3][1][1]} \\\\
    Quadrant 2 & {quadrant_regions[2]} & {phase_data[p1][2][0]:,.2f} & {phase_data[p1][2][1]} & {phase_data[p2][2][0]:,.2f} & {phase_data[p2][2][1]} & {phase_data[p3][2][0]:,.2f} & {phase_data[p3][2][1]} \\\\
    Quadrant 3 & {quadrant_regions[3]} & {phase_data[p1][3][0]:,.2f} & {phase_data[p1][3][1]} & {phase_data[p2][3][0]:,.2f} & {phase_data[p2][3][1]} & {phase_data[p3][3][0]:,.2f} & {phase_data[p3][3][1]} \\\\
    Quadrant 4 & {quadrant_regions[4]} & {phase_data[p1][4][0]:,.2f} & {phase_data[p1][4][1]} & {phase_data[p2][4][0]:,.2f} & {phase_data[p2][4][1]} & {phase_data[p3][4][0]:,.2f} & {phase_data[p3][4][1]} \\\\
    \\midrule
    \\multicolumn{{2}}{{l}}{{\\textbf{{Total Area Under Cultivation}}}} & \\textbf{{{phase_totals[p1]:,.2f}}} & -- & \\textbf{{{phase_totals[p2]:,.2f}}} & -- & \\textbf{{{phase_totals[p3]:,.2f}}} & -- \\\\
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}"""

    print(latex_table)

#%%
soybean_grids={'SPAM2005V3r2_global_A_TA_SOYB_A.tif':"2005to09",
               'spam2010V2r0_global_A_SOYB_A.tif':"2010to20",
               'spam2020_V2r2_global_A_SOYB_A.tif':"2020to25"}
print_the_quadrant_areas(grids=soybean_grids, crop="soybean")

print("=========================================")

arabica_grids={'SPAM2005V3r2_global_A_TA_ACOF_A.tif':"2005to09",
               'spam2010V2r0_global_A_ACOF_A.tif':"2010to20",
               'spam2020_V2r2_global_A_COFF_A.tif':"2020to25"}
print_the_quadrant_areas(grids=arabica_grids, crop="arabica")
