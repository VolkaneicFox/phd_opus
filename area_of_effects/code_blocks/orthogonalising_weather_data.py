#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 17:49:24 2026

@author: darkmage
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')

drive_path=""

#%% making arabica climate_series.csv, the time series for the climate variables of window 5, hist_delta=20 (d20 indicates 20year reference window)

"""
['prec_flux_lowshare_w5d20.csv',
 'prec_flux_upperextent_w5d20.csv',
 'vpd_max_upperextent_w5d20.csv',
 'vpd_max_uppershare_w5d20.csv',
 'prec_flux_uppershare_w5d20.csv',
 'prec_flux_daily_mean_values.csv',
 'vpd_max_daily_mean_values.csv'
"""

prfl_lowshare_d20 = pd.read_csv(drive_path+"/timeseries_csv//prec_flux_lowshare_w5d20.csv")
prfl_lowshare_d20 =prfl_lowshare_d20.set_index("Unnamed: 0")
prfl_lowshare_d20.columns = ['PfLWshd20_ar1', 'PfLWshd20_ar2','PfLWshd20_ar3', 'PfLWshd20_ar4']

prfl_upextent_d20 = pd.read_csv(drive_path+"timeseries_csv/prec_flux_upperextent_w5d20.csv")
prfl_upextent_d20 = prfl_upextent_d20.set_index("Unnamed: 0")
prfl_upextent_d20.columns = ['PfUPmgd20_ar1', 'PfUPmgd20_ar2', 'PfUPmgd20_ar3','PfUPmgd20_ar4']

vdmx_upextent_d20 = pd.read_csv(drive_path+ 'timeseries_csv/vpd_max_upperextent_w5d20.csv')
vdmx_upextent_d20 = vdmx_upextent_d20.set_index("Unnamed: 0")
vdmx_upextent_d20.columns = ['VdUPmgd20_ar1','VdUPmgd20_ar2', 'VdUPmgd20_ar3', 'VdUPmgd20_ar4']

vdmx_upshare_d20 = pd.read_csv(drive_path+ '/timeseries_csv/vpd_max_uppershare_w5d20.csv')
vdmx_upshare_d20 = vdmx_upshare_d20.set_index("Unnamed: 0")
vdmx_upshare_d20.columns = ['VdUPshd20_ar1', 'VdUPshd20_ar2', 'VdUPshd20_ar3', 'VdUPshd20_ar4']

prfl_upshare_d20 = pd.read_csv(drive_path + '/timeseries_csv/prec_flux_uppershare_w5d20.csv')
prfl_upshare_d20 = prfl_upshare_d20.set_index("Unnamed: 0")
prfl_upshare_d20.columns = ['PfUPshd20_ar1', 'PfUPshd20_ar2', 'PfUPshd20_ar3', 'PfUPshd20_ar4']

Pfmn=pd.read_csv(drive_path+"/timeseries_csv//prec_flux_daily_mean_values.csv")
Pfmn=Pfmn.set_index("Unnamed: 0")
Pfmn=Pfmn.rename(columns={'prec_fluxara_pre_area1':'Pfmn_ar1', 
                          'prec_fluxara_pre_area2':'Pfmn_ar2',
                          'prec_fluxara_pre_area3':'Pfmn_ar3', 
                          'prec_fluxara_pre_area4':'Pfmn_ar4'})

Vdmn=pd.read_csv(drive_path+"/timeseries_csv//vpd_max_daily_mean_values.csv")
Vdmn=Vdmn.set_index("Unnamed: 0")
Vdmn=Vdmn.rename(columns={'vpd_maxara_vpd_area1':'Vdmn_ar1', 
                          'vpd_maxara_vpd_area2':'Vdmn_ar2',
                          'vpd_maxara_vpd_area3':'Vdmn_ar3', 
                          'vpd_maxara_vpd_area4':'Vdmn_ar4'})


full_climate_d20 = pd.concat([prfl_lowshare_d20,
                              prfl_upextent_d20,prfl_upshare_d20,
                              vdmx_upextent_d20,
                              vdmx_upshare_d20,Pfmn,Vdmn],axis=1)

full_climate_d20.to_csv(drive_path+"/timeseries_csv//climate_series_d20.csv")

#%% making soybean climate_series.csv, the time series for the climate variables of window 5, hist_delta=20

"""
['prec_flux_lowshare_w5d20.csv',
 'prec_flux_upperextent_w5d20.csv',
 'vpd_max_upperextent_w5d20.csv',
 'vpd_max_uppershare_w5d20.csv',
 'prec_flux_uppershare_w5d20.csv',
 'prec_flux_daily_mean_values.csv',
 'vpd_max_daily_mean_values.csv'
"""
crop="soybean"
prfl_lowshare_d20 = pd.read_csv(drive_path+f"/timeseries_csv/{crop}prec_flux_lowshare_w5d20.csv")
prfl_lowshare_d20 =prfl_lowshare_d20.set_index("Unnamed: 0")
prfl_lowshare_d20.columns = ['PfLWshd20_ar1', 'PfLWshd20_ar2','PfLWshd20_ar3', 'PfLWshd20_ar4']

prfl_upextent_d20 = pd.read_csv(drive_path+f"timeseries_csv/{crop}prec_flux_upperextent_w5d20.csv")
prfl_upextent_d20 = prfl_upextent_d20.set_index("Unnamed: 0")
prfl_upextent_d20.columns = ['PfUPmgd20_ar1', 'PfUPmgd20_ar2', 'PfUPmgd20_ar3','PfUPmgd20_ar4']

vdmx_upextent_d20 = pd.read_csv(drive_path+ f'timeseries_csv/{crop}vpd_max_upperextent_w5d20.csv')
vdmx_upextent_d20 = vdmx_upextent_d20.set_index("Unnamed: 0")
vdmx_upextent_d20.columns = ['VdUPmgd20_ar1','VdUPmgd20_ar2', 'VdUPmgd20_ar3', 'VdUPmgd20_ar4']

vdmx_upshare_d20 = pd.read_csv(drive_path+ f'/timeseries_csv/{crop}vpd_max_uppershare_w5d20.csv')
vdmx_upshare_d20 = vdmx_upshare_d20.set_index("Unnamed: 0")
vdmx_upshare_d20.columns = ['VdUPshd20_ar1', 'VdUPshd20_ar2', 'VdUPshd20_ar3', 'VdUPshd20_ar4']

prfl_upshare_d20 = pd.read_csv(drive_path + f'/timeseries_csv/{crop}prec_flux_uppershare_w5d20.csv')
prfl_upshare_d20 = prfl_upshare_d20.set_index("Unnamed: 0")
prfl_upshare_d20.columns = ['PfUPshd20_ar1', 'PfUPshd20_ar2', 'PfUPshd20_ar3', 'PfUPshd20_ar4']

Pfmn=pd.read_csv(drive_path+f"/timeseries_csv/{crop}prec_flux_daily_mean_values.csv")
Pfmn=Pfmn.set_index("Unnamed: 0")
Pfmn=Pfmn.rename(columns={'prec_fluxsopr_pr_area1':'Pfmn_ar1', 
                          'prec_fluxsopr_pr_area2':'Pfmn_ar2',
                          'prec_fluxsopr_pr_area3':'Pfmn_ar3', 
                          'prec_fluxsopr_pr_area4':'Pfmn_ar4'})

Vdmn=pd.read_csv(drive_path+f"/timeseries_csv/{crop}vpd_max_daily_mean_values.csv")
Vdmn=Vdmn.set_index("Unnamed: 0")
Vdmn=Vdmn.rename(columns={'vpd_maxsovp_vp_area1':'Vdmn_ar1', 
                          'vpd_maxsovp_vp_area2':'Vdmn_ar2',
                          'vpd_maxsovp_vp_area3':'Vdmn_ar3', 
                          'vpd_maxsovp_vp_area4':'Vdmn_ar4'})


full_climate_d20 = pd.concat([prfl_lowshare_d20,
                              prfl_upextent_d20,prfl_upshare_d20,
                              vdmx_upextent_d20,
                              vdmx_upshare_d20,Pfmn,Vdmn],axis=1)

full_climate_d20.to_csv(drive_path+f"/timeseries_csv//{crop}climate_series_d20.csv")

arabia_d20 = pd.read_csv(drive_path+"/timeseries_csv/climate_series_d20.csv")

#%% making arabica climate_series.csv, the time series for the climate variables of window 5, hist_delta=5

climate_datasets = ['vpd_max_uppershare.csv',  ##  VdUP_share
                   'prec_flux_uppershare.csv', ##  PfUP_share
                   'vpd_max_upperextent.csv',  ##  VdUP_mag
                   'prec_flux_lowershare.csv', ##  PfLW_share
                   'prec_flux_upperextent.csv',##  PfUP_mag
             'prec_flux_daily_mean_values.csv',##  Pfmn
             'vpd_max_daily_mean_values.csv']  ##  Vdmn

VdUP_share=pd.read_csv(drive_path+"/timeseries_csv/vpd_max_uppershare.csv")
VdUP_share=VdUP_share.set_index("Unnamed: 0")
VdUP_share=VdUP_share.rename(columns={"vpd_maxara_upp_area1":"VdUPsh_ar1",
                                      "vpd_maxara_upp_area2":"VdUPsh_ar2",
                                      "vpd_maxara_upp_area3":"VdUPsh_ar3",
                                      "vpd_maxara_upp_area4":"VdUPsh_ar4"})

PfUP_share=pd.read_csv(drive_path+"/timeseries_csv/prec_flux_uppershare.csv")
PfUP_share=PfUP_share.set_index("Unnamed: 0")
PfUP_share=PfUP_share.rename(columns={'prec_flux_ara_upp_area1':"PfUPsh_ar1",
                                'prec_flux_ara_upp_area2':"PfUPsh_ar2",
                                'prec_flux_ara_upp_area3':"PfUPsh_ar3", 
                                'prec_flux_ara_upp_area4':"PfUPsh_ar4"})

VdUP_mag=pd.read_csv(drive_path+"timeseries_csv/vpd_max_upperextent.csv")
VdUP_mag=VdUP_mag.set_index("Unnamed: 0")
VdUP_mag=VdUP_mag.rename(columns={'ara_upp_area1':'VdUPmg_ar1', 
                                  'ara_upp_area2':'VdUPmg_ar2', 
                                  'ara_upp_area3':'VdUPmg_ar3', 
                                  'ara_upp_area4':'VdUPmg_ar4'})

PfLW_share=pd.read_csv(drive_path+"timeseries_csv/prec_flux_lowershare.csv")
PfLW_share=PfLW_share.set_index("Unnamed: 0")
PfLW_share=PfLW_share.rename(columns={'prec_flux_ara_low_area1':'PfLWsh_ar1', 
                                      'prec_flux_ara_low_area2':'PfLWsh_ar2',
                                      'prec_flux_ara_low_area3':'PfLWsh_ar3', 
                                      'prec_flux_ara_low_area4':'PfLWsh_ar4'})

PfUP_mag=pd.read_csv(drive_path+"timeseries_csv/prec_flux_upperextent.csv")
PfUP_mag=PfUP_mag.set_index("Unnamed: 0")
PfUP_mag=PfUP_mag.rename(columns={'prec_flux_ara_upp_area1':'PfUPmg_ar1', 
                                  'prec_flux_ara_upp_area2':'PfUPmg_ar2',
                                  'prec_flux_ara_upp_area3':'PfUPmg_ar3', 
                                  'prec_flux_ara_upp_area4':'PfUPmg_ar4'})

Pfmn=pd.read_csv(drive_path+"timeseries_csv/prec_flux_daily_mean_values.csv")
Pfmn=Pfmn.set_index("Unnamed: 0")
Pfmn=Pfmn.rename(columns={'prec_fluxara_pre_area1':'Pfmn_ar1', 
                          'prec_fluxara_pre_area2':'Pfmn_ar2',
                          'prec_fluxara_pre_area3':'Pfmn_ar3', 
                          'prec_fluxara_pre_area4':'Pfmn_ar4'})

Vdmn=pd.read_csv(drive_path+"timeseries_csv/vpd_max_daily_mean_values.csv")
Vdmn=Vdmn.set_index("Unnamed: 0")
Vdmn=Vdmn.rename(columns={'vpd_maxara_vpd_area1':'Vdmn_ar1', 
                          'vpd_maxara_vpd_area2':'Vdmn_ar2',
                          'vpd_maxara_vpd_area3':'Vdmn_ar3', 
                          'vpd_maxara_vpd_area4':'Vdmn_ar4'})

full_climate = pd.concat([VdUP_share,PfUP_share,
                          VdUP_mag,PfLW_share,
                          PfUP_mag,Pfmn,Vdmn],axis=1)

full_climate.to_csv(drive_path+"timeseries_csv/climate_series.csv")

#%% arranging index in climate_series.csv, for window5, delta5
climate=pd.read_csv(f"{drive_path}/climate_series.csv") ## window 5, historical delta 5, and the means of VPD and Prec Flux
climate["mm-dd"]=climate["Unnamed: 0"].apply(lambda x: x[5:])
climate=climate[climate["mm-dd"]!="02-29"]
climate=climate.set_index("Unnamed: 0")
climate=climate.sort_index()
print(f"The number of years in view = {climate.shape[0]/365}")
"""
first adjustment: removing the leap year observations since they did not 
have the same reference length as the other days, also bad juju on  leap
years

The number of years in view = 21.0
"""
print(climate.columns)
print("Shape = ",climate.shape)
"""
Index(['VdUPsh_ar1', 'VdUPsh_ar2', 'VdUPsh_ar3', 'VdUPsh_ar4',
       'PfUPsh_ar1', 'PfUPsh_ar2', 'PfUPsh_ar3', 'PfUPsh_ar4', 'VdUPmg_ar1',
       'VdUPmg_ar2', 'VdUPmg_ar3', 'VdUPmg_ar4', 'PfLWsh_ar1', 'PfLWsh_ar2',
       'PfLWsh_ar3', 'PfLWsh_ar4', 'PfUPmg_ar1', 'PfUPmg_ar2', 'PfUPmg_ar3',
       'PfUPmg_ar4', 'Pfmn_ar1', 'Pfmn_ar2', 'Pfmn_ar3', 'Pfmn_ar4',
       'Vdmn_ar1', 'Vdmn_ar2', 'Vdmn_ar3', 'Vdmn_ar4', 'mm-dd'],
      dtype='str')
Start Date = 2005-01-01
End Date = 2025-12-31
Shape =  (7665, 29)
"""
#%% 
"""
making three different dummies:
    1. the year dummies
    2. the month dummies
beginning by orthogonalising each feature on year and month separate dummies,
and then on the month-year dummies
for the measure of breach magnitude, as for VdUPmg* and PfUPmg*, this is relative
to the threshold value, and each threshold value is generated from the +/-5 days 
band around the given date
"""

def array_projection_matrix(climate_ar,ndays,nyears,projection="both"):   
    csum_days_in_month=np.cumsum(np.array([0,31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]))
    day = np.arange(365)[:, None]                                   # (365, 1)
    month_dummies = ((day >= csum_days_in_month[:-1]) &
                     (day <  csum_days_in_month[1:])).astype(float)  # (365, 12)
    month_dummies = np.vstack([month_dummies]*nyears) ## (7665,12)
    print(f"{month_dummies.shape[0] == climate_ar.shape[0]}") ## True
    csum_days_in_year=np.arange(0,climate_ar.shape[0]+1,step=365)
    day=np.arange(climate_ar.shape[0])[:,None]
    year_dummies = ((day >= csum_days_in_year[:-1]) &
                     (day <  csum_days_in_year[1:])).astype(float) # (7665,21)
    print(f"{year_dummies.shape[0]==climate_ar.shape[0]}") #True
    if projection=="month": 
        all_dummies=month_dummies
    elif projection=="year":
        all_dummies=year_dummies
    elif projection=="month and year":
        None ## 12 *21 from month_dummies before vstack
    elif projection=="both":
        all_dummies = np.hstack([month_dummies,year_dummies]) ## (7665,33)
    coef = np.linalg.solve(all_dummies.T @ all_dummies, all_dummies.T @ climate_ar)  # 7665X12
    climate_ar_tilde = climate_ar - all_dummies @ coef

    return climate_ar_tilde


#%% orthogonal climate series for window 5, historical delta = 5
climate=pd.read_csv(drive_path+"/climate_series.csv")
climate["mm-dd"]=climate["Unnamed: 0"].apply(lambda x: x[5:])
climate=climate[climate["mm-dd"]!="02-09"]
climate=climate.drop("mm-dd",axis=1)
climate=climate.set_index("Unnamed: 0")
nyears,ndays=climate.shape[0]//365, 365
climate_ar = climate.drop(columns="mm-dd").to_numpy()
climate_ar_tilde=array_projection_matrix(climate_ar)

climate_df_tilde=pd.DataFrame(index=climate.index,
                              columns=climate.drop("mm-dd",axis=1).columns,
                              data=climate_ar_tilde)

climate_df_tilde.to_csv(f"{drive_path}/climate_orthogonalised.csv")

#%% orthogonal climate series for window 5, historical delta = 20

climate_d20=pd.read_csv(drive_path+"/climate_series_d20.csv")
climate_d20["mm-dd"]=climate_d20["Unnamed: 0"].apply(lambda x: x[5:])
climate_d20=climate_d20[climate_d20["mm-dd"]!="02-29"]
climate_d20 = climate_d20.drop("mm-dd",axis=1)
climate_d20=climate_d20.set_index("Unnamed: 0")
nyears,ndays=climate_d20.shape[0]//365, 365
climate_ar = climate_d20.to_numpy()
climate_ar_tilde=array_projection_matrix(climate_ar,ndays=ndays,nyears=nyears)

climate_df_tilde=pd.DataFrame(index=climate_d20.index,
                              columns=climate_d20.columns,
                              data=climate_ar_tilde)
climate_df_tilde.to_csv(f"{drive_path}/climate_orthogonalised_d20.csv")
"""
climate_d20.columns
Out[9]: 
Index(['PfLWshd20_ar1', 'PfLWshd20_ar2', 'PfLWshd20_ar3', 'PfLWshd20_ar4',
       'PfUPmgd20_ar1', 'PfUPmgd20_ar2', 'PfUPmgd20_ar3', 'PfUPmgd20_ar4',
       'PfUPshd20_ar1', 'PfUPshd20_ar2', 'PfUPshd20_ar3', 'PfUPshd20_ar4',
       'VdUPmgd20_ar1', 'VdUPmgd20_ar2', 'VdUPmgd20_ar3', 'VdUPmgd20_ar4',
       'VdUPshd20_ar1', 'VdUPshd20_ar2', 'VdUPshd20_ar3', 'VdUPshd20_ar4',
       'Pfmn_ar1', 'Pfmn_ar2', 'Pfmn_ar3', 'Pfmn_ar4', 'Vdmn_ar1', 'Vdmn_ar2',
       'Vdmn_ar3', 'Vdmn_ar4'],
      dtype='str')
"""

#%% orthogonal climate series for window 5, historical delta = 5
## NOT DONE YET!!!

crop="soybean"
climate=pd.read_csv(drive_path+f"/{crop}climate_series.csv")
climate["mm-dd"]=climate["Unnamed: 0"].apply(lambda x: x[5:])
climate=climate[climate["mm-dd"]!="02-09"]
climate=climate.drop("mm-dd",axis=1)
climate=climate.set_index("Unnamed: 0")
nyears,ndays=climate.shape[0]//365, 365
climate_ar = climate.drop(columns="mm-dd").to_numpy()
climate_ar_tilde=array_projection_matrix(climate_ar)

climate_df_tilde=pd.DataFrame(index=climate.index,
                              columns=climate.drop("mm-dd",axis=1).columns,
                              data=climate_ar_tilde)

climate_df_tilde.to_csv(f"{drive_path}/{crop}climate_orthogonalised.csv")

#%% orthogonal climate series for window 5, historical delta = 20

crop="soybean"
climate_d20=pd.read_csv(drive_path+f"/timeseries_csv/{crop}climate_series_d20.csv")
climate_d20["mm-dd"]=climate_d20["Unnamed: 0"].apply(lambda x: x[5:])
climate_d20=climate_d20[climate_d20["mm-dd"]!="02-29"]
climate_d20 = climate_d20.drop("mm-dd",axis=1)
climate_d20=climate_d20.set_index("Unnamed: 0")
nyears,ndays=climate_d20.shape[0]//365, 365
climate_ar = climate_d20.to_numpy()
climate_ar_tilde=array_projection_matrix(climate_ar,ndays=ndays,nyears=nyears)

climate_df_tilde=pd.DataFrame(index=climate_d20.index,
                              columns=climate_d20.columns,
                              data=climate_ar_tilde)
climate_df_tilde.to_csv(f"{drive_path}/timeseries_csv/{crop}climate_orthogonalised_d20.csv")
