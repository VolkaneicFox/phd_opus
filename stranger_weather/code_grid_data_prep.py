distr# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 08:59:06 2023

@author: nb
"""

import numpy as np
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pandas as pd
import hvplot.pandas
import os, pickle
import matplotlib.colors as mcolors
import matplotlib.cm as cm


from shapely.geometry import Point
drive='D:'
ncdf=xr.open_dataset(drive+'\\weather_grid_data\\tas_raw_grid_DE.nc')
ncdf_present=ncdf.sel(time=ncdf.time.dt.year>=1990)

x_list,y_list=np.arange(0,ncdf.x.shape[0],1),np.arange(0,ncdf.y.shape[0],1)
coords_list=[[x,y] for x in x_list for y in y_list]

## url:https://gdz.bkg.bund.de/index.php/default/verwaltungsgebiete-1-5-000-000-stand-31-12-vg5000-12-31.html
import geopandas as gpd
drive='D:'
kreise_file=gpd.read_file('G:\\SME\Paper_1\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_crs=kreise_file.to_crs(epsg=3034)
kreise_crs=kreise_crs[['Kreise_Code','geometry']]
kreise_list=list(kreise_crs['Kreise_Code'])
kreise_crs=kreise_crs.set_index('Kreise_Code')

kreise_file1=gpd.read_file("G://SME/Paper_1/vg5000_12_31_gk3_shape_ebenen/vg5000_ebenen_1231/VG5000_KRS.shp",crs=4326)
kreise_file1['Kreise_Code']=kreise_file1['ARS'].astype(int)
kreise_crs1=kreise_file1.to_crs(epsg=3034)
kreise_crs1=kreise_crs1[['Kreise_Code','geometry']]
kreise_list1=list(kreise_crs1['Kreise_Code'])
kreise_crs1=kreise_crs1.set_index('Kreise_Code')

## step 1: select the coordinates in the ncdf that are from the (multi)polygon 
## geometry of the kreies_crs shapefile
'''
kreise_coords={}
ix=0
for kreise in kreise_list:
    kreise_geo=kreise_crs.loc[kreise,'geometry']
    found={'coords_3034':[],
           'xy_axis':[]}
    for coords in coords_list:
        j,i=coords[0],coords[1]
        make_point=(ncdf.x.values[j],ncdf.y.values[i]) #<-- finding the points that are in the ncdf
        if kreise_geo.contains(Point(make_point)) or kreise_geo.touches(Point(make_point)):
            found['coords_3034']=found['coords_3034']+[[int(make_point[0]),int(make_point[1])]]#<-- the points in the ncdf that correspond to the polygon in the shapefile
            found['xy_axis']=found['xy_axis']+[[j,i]]
    kreise_coords[kreise]=found
    print(str(ix)+': '+str(kreise))
    ix=ix+1[]
'''
import pickle
with open('G://SME/Paper_1/kreise_coords_tas.pickle','rb') as infile:
    kreise_coords=pickle.load(infile)
size_list={item:len(kreise_coords[item]['xy_axis']) for item in kreise_coords}
c_small=[item for item in kreise_coords if len(kreise_coords[item]['xy_axis'])<50]

c_mid1=[item for item in kreise_coords if len(kreise_coords[item]['xy_axis'])>=50]
c_mid1=[item for item in c_mid1 if len(kreise_coords[item]['xy_axis'])<70]
c_mid1=[i for i in c_mid1 if i!=1055]

c_mid2=[item for item in kreise_coords if len(kreise_coords[item]['xy_axis'])>=70]
c_mid2=[item for item in c_mid2 if len(kreise_coords[item]['xy_axis'])<90]

c_large=[item for item in kreise_coords if len(kreise_coords[item]['xy_axis'])>=90]

   
#%% ##step 2: Making the record of daily observations at the county level

'''
a. select the grids of the county
b. assign the weights
c. take daily observation 
'''


ix=1
done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
left=[item for item in kreise_coords if str(item)+'.txt' not in done]    

#for console 1
console_1=left[1:17]
for kreise in console_1:
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1



##for console 2

ix=1
console_2=left[17:33]
for kreise in console_2:
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1
        
##for console 3

ix=1
done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
left=[item for item in kreise_coords if str(item)+'.txt' not in done]
console_3=left[33:49]    
for kreise in console_3:
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1

##for console 4

ix=1
done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
left=[item for item in kreise_coords if str(item)+'.txt' not in done]
console_4=left[49:]  
for kreise in console_4:
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1

##for console small 
ix=1
for kreise in c_small:
    done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1
        
##for console c_mid1 
ix=1
consoles=console_2+console_3+console_4
done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
left=[item for item in kreise_coords if str(item)+'.txt' not in done]
c_mid1=[item for item in left if item not in consoles] 
for kreise in c_mid1[:10]:
    done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1
        
##for console c_mid2 
ix=1

for kreise in c_mid1[10:20]:
    done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1

##for console c_large
ix=1
for kreise in c_mid1[20:]:
    done=os.listdir(drive+'//weather_grid_data/kreise_grid_daily/')
    if str(kreise)+'.txt' not in done:
        print('Here with: ',kreise)
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        weights = np.cos(np.deg2rad(kreise_ncdf.lat))
        weights.name = "weights"
        kreise_ncdf_weighted=kreise_ncdf.tas.weighted(weights)
        kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
        kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
        kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
        kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
        kreise_ncdf_pd.to_csv(drive+'\\weather_grid_data\\kreise_grid_daily\\'+str(kreise)+'.txt')
        print(ix,': ',kreise)
        ix=ix+1


'''
grid_weights=np.zeros((x_list.shape[0],y_list.shape[0]))
for coords in coords_list:
    lat_val=ncdf.lat[coords[0],coords[1]].values
    grid_weights[coords[0]][coords[1]]=np.cos(np.deg2rad(lat_val))
'''

#%% checking correlation of TMK_hgt+hav and TAS
paths={'tmk':'D:\\weather_data_Sept2023\\kreise_tmk_daily_wt\\',
       'tas':'D:\\weather_grid_data\\kreise_grid_daily\\'}

kreise_files=os.listdir(paths['tas'])

corr_df=pd.DataFrame()

for file in kreise_files:
    tasdf=pd.read_csv(paths['tas']+file)[['MESS_DATUM','tas']]
    tmkdf=pd.read_csv(paths['tmk']+file)[['MESS_DATUM','TMK_hgt+hav']]
    tasdf=pd.merge(tmkdf,tasdf,on='MESS_DATUM',how='inner')
    corr_df.loc[int(file.split('.')[0]),'corr_coef']=tasdf['TMK_hgt+hav'].corr(tasdf['tas'])

fig,ax=plt.subplots()
ax.hist(corr_df['corr_coef'])
ax.set_title('Histogram of Correlation Coefficient between Grid Data and Station Data')
plt.show()

#%% Making annual measures of climate data seasonally adjusted %%#

### TEMPERATURE ###
'''
TAS:
    > we have the seasonally adjusted mean tas, tasq3_sna, std
      days count, 
    > we need non-seasonally adjusted mean tas, std
'''      

def findpercentile(indf,flag):
    flaghead=flag+'_2020'
    indf['u99'+flaghead]=0
    indf['l01'+flaghead]=0
    indf['hotdays99'+flaghead]=0
    indf['colddays01'+flaghead]=0
    for month in range(12):
        month=month+1
        indf['month_d']=0
        indf['month_d']=(indf['month']==month).astype(int)
        distbn=indf[indf['month']==month][['MESS_DATUM',flag]]
        distbn=distbn[flag]
        distbn=sorted(distbn)
        indf['u99'+flaghead]=indf['u99'+flaghead]+(indf['month_d']*distbn[int(0.99*len(distbn))])
        indf['l01'+flaghead]=indf['l01'+flaghead]+(indf['month_d']*distbn[int(0.01*len(distbn))])
        indf['hotdays99'+flaghead]=indf['hotdays99'+flaghead]+(indf['month_d']*(indf[flag]>=indf['u99'+flaghead]).astype(int))
        indf['colddays01'+flaghead]=indf['colddays01'+flaghead]+(indf['month_d']*(indf[flag]<=indf['l01'+flaghead]).astype(int))
    indf=indf.drop('month_d',axis=1)
    return indf
  
allpath="D:\\weather_grid_data\\kreise_grid_daily_tas\\"

flag='tas'
cols=['year','Kreise_Code', 'tas','tas_q3', 'hotdays99tas_2010','hotdays99tas_2020',
      'colddays01tas_2010','colddays01tas_2020']
tasdf=pd.DataFrame()
#kreise_files=os.listdir(allpath)
for file in ['3159.txt']:
    tempfile=pd.read_csv(allpath+file)
    tempfile=tempfile.drop([i for i in tempfile.columns if '2010' in i])
    #tempfile=tempfile[tempfile['MESS_DATUM']>=19900100]
    #tempfile['yyyymm']=(tempfile['MESS_DATUM']/100).astype(int)
    #tempfile['month']=(tempfile['yyyymm']%100).astype(int)
    #tempfile['year']=(tempfile['yyyymm']/100).astype(int)
    years=list(set(tempfile['year']))
    #tempfile['Kreise_Code']=int((file.split('.')[0]))
    #tempfile2010=findpercentile(tempfile,flag=flag,hist=True)
    tempfile2020=findpercentile(tempfile,flag=flag)
    tempfile2020.to_csv('D://weather_grid_data//test3159.csv',index=False)
    #tempfile.to_csv(allpath+file,index=False)
    #tempdf=tempfile[['yyyymm','tas']].groupby(by='yyyymm').mean()
    #tempdf=tempdf.reset_index()
    #tempdf['year']=(tempdf['yyyymm']/100).astype(int)
    #tempdf=tempdf[['year','tas']].groupby(by='year').mean()
    #months=list(set(tempfile['month']))
    #tempdf=pd.DataFrame(index=years)
    #tempdf['Kreis_Code']=int(file.split('.')[0])
    #tempdf['tasq3_sna']=0
    #for year in years:
    #    tas_list=[]
    #    yearfile=tempfile[tempfile['year']==year]
    #    for month in months :
    #        temp_list=yearfile[yearfile['month']==month]['tas'].sort_values().reset_index(drop=True)
    #        tas_list=tas_list+list(temp_list.loc[int(0.50*31):int(0.75*31)])
    #    tempdf.loc[year,'tasq3_sna']=np.mean(tas_list)
    #tempdf=tempdf.reset_index()
    #tempdf=tempdf.rename(columns={'index':'year'})
    #daysdf=tempfile[['year',
    #                 'hotdays99tas_2010','colddays01tas_2010',
    #                 'hotdays99tas_2020','colddays01tas_2020']]
    #daysdf=daysdf.groupby(by='year').sum().reset_index()
    #daysdf['Kreise_Code']=int(file.split('.')[0])     
    #tempdf=pd.merge(tempdf,daysdf,on='year',how='inner')
    #tempdf['Kreise_Code']=int(file.split('.')[0])
    #tempdf=tempdf.reset_index(drop=True)
    #tasdf=pd.concat([tasdf,tempdf],axis=0)
    
## measuring annual variability in temperatures , seasonally adjusted

allpath='D:\\weather_data_Sept2023\\kreise_tmk_daily_wt\\'
files_list=os.listdir(allpath)
flag='TMK_hgt+hav'
years=[1995+i for i in range(26)]
months=[i+1 for i in range(12)]
var_df2=pd.DataFrame()
for file in files_list:
    infile=pd.read_csv(allpath+file)
    tempdf=pd.DataFrame()
    tempdf[flag+'_q3_sna']=0
    tempdf[flag+'_q3_nsna']=0
    for year in years:
        tas_list=[]
        tas_nsnalist=[]
        yearfile=infile[infile['year']==year]
        for month in months :
            temp_list=yearfile[yearfile['month']==month][flag].sort_values().reset_index(drop=True)
            tas_list=tas_list+list(temp_list.loc[int(0.50*31):int(0.75*31)])
        tempdf.loc[year,flag+'_q3_sna']=np.mean(tas_list)
        temp_list_nsna=yearfile[flag].sort_values().reset_index(drop=True)
        tas_nsna=temp_list_nsna.loc[int(0.50*366):int(0.75*366)].mean()
        tempdf.loc[year,flag+'_q3_nsna']=tas_nsna
    tempdf['Kreise_Code']=int(file.split('.')[0])
    tempdf=tempdf.reset_index()
    tempdf=tempdf.rename(columns={'index':'year'})
    var_df2=pd.concat([var_df2,tempdf],axis=0)
    
## measuring non-seasonally adjusted mean and std in temperatures

var_df3=pd.DataFrame()
for file in files_list:
    infile=pd.read_csv(allpath+file)
    meanfile=infile[[flag,'year']].groupby(by='year').mean().reset_index()
    stdfile=infile[[flag,'year']].groupby(by='year').std().reset_index()
    meanfile=meanfile.rename(columns={flag:flag+'_nsna'})
    stdfile=stdfile.rename(columns={flag:flag+'_std_nsna'})
    meanfile['Kreise_Code']=int(file.split('.')[0])
    stdfile['Kreise_Code']=int(file.split('.')[0])
    temp=pd.merge(meanfile,stdfile,on=['year','Kreise_Code'],how='inner')
    stddf=infile[[flag,'yyyymm']].groupby(by='yyyymm').std().reset_index()
    stddf['year']=(stddf['yyyymm']/100).astype(int)
    stddf=stddf[['year',flag]].groupby(by='year').mean().reset_index()
    stddf=stddf.rename(columns={flag:flag+'_std_sna'})
    stddf['Kreise_Code']=int(file.split('.')[0])
    temp=pd.merge(temp,stddf,on=['Kreise_Code','year'],how='inner')
    meandf=infile[[flag,'yyyymm']].groupby(by='yyyymm').mean().reset_index()
    meandf['year']=(meandf['yyyymm']/100).astype(int)
    meandf=meandf[['year',flag]].groupby(by='year').mean().reset_index()
    meandf['Kreise_Code']=int(file.split('.')[0])
    meandf=meandf.rename(columns={flag:flag+'_sna'})
    temp=pd.merge(temp,meandf,on=['year','Kreise_Code'],how='inner')
    var_df3=pd.concat([var_df3,temp],axis=0)

#var_df=pd.merge(var_df,var_df1,on=['year','Kreise_Code'],how='inner')
#%% combining with economic data

tasdf.to_csv('D:\\weather_grid_data\\annual_tas.csv',index=False)
stn_full=pd.read_csv('D:\\weather_data_Sept2023\\modelDFSeptv2.csv')
stn_full['Kreise_Code']=stn_full['Kreise_Code'].replace(3152,3159)
unify=stn_full[['Kreise_Code','West-Germany','East-Germany']].drop_duplicates()
econdf=pd.read_csv('D:\\weather_data_Sept2023\\anEcon_kreise.csv')
econdf['Kreise_Code']=econdf['Kreise_Code'].replace(3152,3159)
fulldf=pd.merge(econdf,tasdf,on=['year','Kreise_Code'],how='inner')
fulldf=pd.merge(fulldf,unify,on='Kreise_Code',how='inner')
fulldf.to_csv('D:\\weather_grid_data\\modelVSept2023.csv',index=False)

infull=pd.read_csv('D:\\weather_grid_data\\model_Octv4.csv')
infull=pd.merge(infull,var_df1,on=['year','Kreise_Code'],how='inner')
infull.to_csv('D:\\weather_grid_data\\model_Octv4.csv',index=False)

#%% comparing tmk_hgt+hav and tas annual data
path_tmk='D:\\weather_data_Sept2023\\modelDFSeptv2.csv'
path_tas='D:\\weather_grid_data\\modelVSept2023.csv'

tmk_df=pd.read_csv(path_tmk)
tmk_df['Kreise_Code']=tmk_df['Kreise_Code'].replace(3152,3159)
tas_df=pd.read_csv(path_tas)
check=pd.merge(tmk_df,tas_df, on=['Kreise_Code','year'],how='inner')
check['diff_abs']=abs(check['TMK_hgt+hav']-check['tas'])
check['diff']=check['TMK_hgt+hav']-check['tas']

#%% taking measures of variability for TMK_hgt+hav and tas

##1. combining all daily level observations into a single dataframe

path_tmk='D:\\weather_data_Sept2023\\kreise_tmk_daily_wt\\'
path_tas='D:\\weather_grid_data\\kreise_grid_daily_tas\\'

tmk_df=pd.DataFrame()
tas_df=pd.DataFrame()

for file in os.listdir(path_tas):
    tmkin=pd.read_csv(path_tmk+file)
    tmk_df=pd.concat([tmk_df,tmkin],axis=0)
    tasin=pd.read_csv(path_tas+file)
    tas_df=pd.concat([tas_df,tasin],axis=0)


#%% compiling sectoral GVA growrth rates with the TMK_hgt+hav and tas measures
path='D:\\archive\\Panel_Modelling\\'
files={'svc_sector.csv':'svc_GVA','primary_sector.csv':'primary_GVA','man_sector.csv':'man_GVA','popn.csv':'popn'}
maincols=['Kreise_Code','year']
sector_df1=pd.DataFrame(columns=maincols)
for file in files:
    infile=pd.read_csv(path+file)[maincols+[files[file]]]
    #kreise_list=list(set(infile['Kreise_Code']))
    infile['Kreise_Code']=infile['Kreise_Code'].replace(3152,3159)
    outfile=pd.DataFrame()
    for kreise in kreise_list:
        temp=infile[infile['Kreise_Code']==kreise]
        temp=temp.set_index('year')
        temp[files[file]+'ln_diff1']=np.log(temp[files[file]]).diff(1)
        temp=temp.reset_index()
        outfile=pd.concat([outfile,temp],axis=0)
    sector_df1=pd.merge(sector_df1,outfile,on=['year','Kreise_Code'],how='outer')
        
        
        

#%% Gathering and compiling precipitation data

drive='D:'
kreise_file=gpd.read_file(drive+'\\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_crs=kreise_file.to_crs(epsg=3034)
kreise_crs=kreise_crs[['Kreise_Code','geometry']]
kreise_list=list(kreise_crs['Kreise_Code'])
kreise_crs=kreise_crs.set_index('Kreise_Code')

ncdf=xr.open_dataset('D:\\weather_grid_data\\pr_raw_grid_DE.nc')
ncdf_present=ncdf.sel(time=ncdf.time.dt.year>=1930)

x_list,y_list=np.arange(0,ncdf.x.shape[0],1),np.arange(0,ncdf.y.shape[0],1)
coords_list=[[x,y] for x in x_list for y in y_list]



def kreise_find(kreise):
    kreise_geo=kreise_crs.loc[kreise,'geometry']
    found={'coords_3034':[],
           'xy_axis':[]}
    for coords in coords_list:
        j,i=coords[0],coords[1]
        make_point=(ncdf.x.values[j],ncdf.y.values[i]) #<-- finding the points that are in the ncdf
        if kreise_geo.contains(Point(make_point)) or kreise_geo.touches(Point(make_point)):
            found['coords_3034']=found['coords_3034']+[[int(make_point[0]),int(make_point[1])]]#<-- the points in the ncdf that correspond to the polygon in the shapefile
            found['xy_axis']=found['xy_axis']+[[j,i]]
    return found
    

for kreise in kreise_list[11:101]:
    kreise_coords[kreise]=kreise_find(kreise)
    
#console 2
kreise_coords2={}
for kreise in kreise_list[101:151]:
    kreise_coords2[kreise]=kreise_find(kreise)

#console 3
kreise_coords3={}
for kreise in kreise_list[151:201]:
    kreise_coords3[kreise]=kreise_find(kreise)
    
#console 4
kreise_coords41={}
for kreise in kreise_list[301:351]:
    kreise_coords41[kreise]=kreise_find(kreise)
    
#console 5
kreise_coords51={}
for kreise in kreise_list[351:]:
    kreise_coords51[kreise]=kreise_find(kreise)
        
kreise_coords_all=['','2','3','4','41','51','5']
kreise_coords_full={}
for num in kreise_coords_all:
    with open('C://Users//nb//Desktop//kreise_coords'+num+'.pickle', 'rb') as handle:
        b = pickle.load(handle)
        for item in b:
            kreise_coords_full[item]=b[item]
    
with open('D://weather_grid_data//kreise_coords_pr.pickle', 'wb') as handle:
    pickle.dump(kreise_coords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    



'''
def catch_rain(kreise):
    inkreise=kreise_coords[kreise]['xy_axis']
    kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
    weights = np.cos(np.deg2rad(kreise_ncdf.lat))
    weights.name = "weights"
    kreise_ncdf_weighted=kreise_ncdf.pr.weighted(weights)
    kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
    kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
    kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
    return kreise_ncdf_pd

console1,console2,console3,console4,console5=kreise_list[:51],kreise_list[51:101],kreise_list[101:151],kreise_list[151:201],kreise_list[201:251]
console6,console7,console8=kreise_list[251:301],kreise_list[301:351],kreise_list[351:]

'''
import pandas as pd
import numpy as np
import os, pickle
import xarray as xr

with open('G://SME/Paper_1/kreise_coords_pr.pickle', 'rb') as handle:
    kreise_coords=pickle.load(handle)
kreise_list=list(kreise_coords.keys())
ncdf=xr.open_dataset('G://SME/Paper_1/pr_hyras_v5DE.nc')
ncdf_present=ncdf.sel(time=ncdf.time.dt.year>=1950)

ix=0
#def catch_rain(kreise):
kreise=5958
inkreise=kreise_coords[kreise]['xy_axis']
kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
kreise_ncdf_df=pd.DataFrame(index=range(kreise_ncdf.time.values.shape[0]))
weights=np.cos(np.deg2rad(kreise_ncdf.lat))
weights.name = "weights"
kreise_ncdf_weighted=kreise_ncdf.pr.weighted(weights)
kreise_ncdf_fin=kreise_ncdf_weighted.mean(('y','x'))
kreise_ncdf_pd=kreise_ncdf_fin.to_dataframe()
kreise_ncdf_pd=kreise_ncdf_pd.reset_index()
kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['time'].apply(lambda x: str(x).split(' ')[0])
kreise_ncdf_pd['MESS_DATUM']=kreise_ncdf_pd['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
kreise_ncdf_pd.to_csv("G://SME/Paper_1/kreise_daily_pr_full/"+str(kreise)+".txt",index=False)
#    print(ix,': ',kreise)
#        ix=ix+1
#        return kreise_ncdf_pd

coords_size={item:len(kreise_coords[item]['xy_axis']) for item in kreise_coords}
small1_kreise=[item for item in coords_size if coords_size[item]<=250]
small2_kreise=[item for item in coords_size if (coords_size[item]>250)&(coords_size[item]<=750)]
small3_kreise=[item for item in coords_size if (coords_size[item]>750)&(coords_size[item]<=1250)]
med1_kreise=[item for item in coords_size if (coords_size[item]>1250)&(coords_size[item]<=1750)]
med2_kreise=[item for item in coords_size if (coords_size[item]>1750)&(coords_size[item]<2000)]
large3_kreise=[item for item in coords_size if (coords_size[item]>=2000)]

        
for kreise in small1_kreise:
        console1df=catch_rain(kreise)

for kreise in small2_kreise:
        console1df=catch_rain(kreise)
        
for kreise in small3_kreise:
        console1df=catch_rain(kreise)
        
for kreise in med1_kreise:
        console1df=catch_rain(kreise)
        
for kreise in med2_kreise:
        console1df=catch_rain(kreise)
        
for kreise in large3_kreise:
        console1df=catch_rain(kreise)
        
small3_kreise=[item for item in left_79_size if left_79_size[item]>=1201 and left_79_size[item]<=1500]

##HERE##
for kreise in [1058]:
    #all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
    #if (str(kreise)+'.txt' in all_files) == False:
    print(kreise)
    console2df=catch_rain(kreise)
    consolde2df=console2df[['MESS_DATUM','prec']]
    console2df['Kreise_Code']=kreise
    console2df.to_csv('D:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
    #ix=ix+1

  
ix=1

coords_size={item:len(kreise_coords2[item]['xy_axis']) for item in kreise_coords2}
med1_kreise=[item for item in left_79_size if left_79_size[item]>=1501 and left_79_size[item]<=1700]
for kreise in [8426, 12070, 14522, 14523, 14625, 14626]:
    #all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
    if (str(kreise)+'.txt' in all_files) == False:
        print(kreise)
        console3df=catch_rain(kreise)
        consolde3df=console3df[['MESS_DATUM','prec']]
        console3df['Kreise_Code']=kreise
        console3df.to_csv('E:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
    ix=ix+1
    
ix=1

coords_size={item:len(kreise_coords2[item]['xy_axis']) for item in kreise_coords2}
med2_kreise=[item for item in left_79_size if left_79_size[item]>=1701 and left_79_size[item]<=2000]
for kreise in [14730, 15081, 15083, 15085, 15090, 15091]:
    #all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
    if (str(kreise)+'.txt' in all_files) == False:
        print(kreise)
        console4df=catch_rain(kreise)
        consolde4df=console4df[['MESS_DATUM','prec']]
        console4df['Kreise_Code']=kreise
        console4df.to_csv('E:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
    ix=ix+1
    
ix=1
all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
coords_size={item:len(kreise_coords[item]['xy_axis']) for item in kreise_coords}
med3_kreise=[item for item in coords_size if coords_size[item]>=2001 and  coords_size[item]<=2500 ]
for kreise in 16063, 16066, 16075, 9372, 9571]:
    all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
    if (str(kreise)+'.txt' in all_files) == False:
        print(kreise)
        console5df=catch_rain(kreise)
        consolde5df=console5df[['MESS_DATUM','prec']]
        console5df['Kreise_Code']=kreise
        console5df.to_csv('E:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
        ix=ix+1
    
    
ix=1

coords_size={item:len(kreise_coords[item]['xy_axis']) for item in kreise_coords}
large1_kreise=[item for item in coords_size if coords_size[item]>=2501 and  coords_size[item]<=4000 ]
for kreise in [14522,14626,14730,15085,15091,9571]:
    all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
    if (str(kreise)+'.txt' in all_files) == False:
        print(kreise)
        console6df=catch_rain(kreise)
        consolde6df=console6df[['MESS_DATUM','prec']]
        console6df['Kreise_Code']=kreise
        console6df.to_csv('E:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
    ix=ix+1
    
ix=1

coords_size={item:len(kreise_coords[item]['xy_axis']) for item in kreise_coords}
large2_kreise=[item for item in coords_size if coords_size[item]>=4001]
for kreise in [14627,15082,15084,15086,15087,15088,15089,16063,16066,16075]:
    all_files=os.listdir('E:\\weather_grid_data\\kreise_daily_pr\\')
    if (str(kreise)+'.txt' in all_files) == False:
        print(kreise)
        console7df=catch_rain(kreise)
        consolde7df=console7df[['MESS_DATUM','prec']]
        console7df['Kreise_Code']=kreise
        console7df.to_csv('E:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
    ix=ix+1


def catch_rain(kreise):
        inkreise=kreise_coords[kreise]['xy_axis']
        kreise_ncdf=ncdf_present.isel(x=[cd[0] for cd in inkreise],y=[cd[1] for cd in inkreise])
        kreise_ncdf_df=pd.DataFrame(index=range(kreise_ncdf.time.values.shape[0]))
        weights=np.cos(np.deg2rad(kreise_ncdf.lat)).values
        kreise_ncdf_df['prec']=nan
        kreise_ncdf_df['MESS_DATUM']=nan
        for i in range(kreise_ncdf.time.values.shape[0]):
            time_stamp=kreise_ncdf.time.values[i]
            mess_datum=str(time_stamp).split('T')[0]
            print('Taking date and starting weights')
            pr=kreise_ncdf.sel(time=time_stamp).pr.values
            pr=np.nan_to_num(pr)
            pr=item_mult(pr,weights)
            print('weights done')
            pr=item_mult(pr,weights).mean()
            kreise_ncdf_df.loc[i,'MESS_DATUM']=mess_datum
            kreise_ncdf_df.loc[i,'prec']=pr
            kreise_ncdf_df.loc[i,'MESS_DATUM']=mess_datum
            kreise_ncdf_df.loc[i,'prec']=pr
            print('ix:' ,i)
        return kreise_ncdf_df
    
    
ix=1
for kreise in [9677]:
    print(ix)
    console8df=catch_rain(kreise)
    consolde8df=console8df[['MESS_DATUM','prec']]
    console8df['Kreise_Code']=kreise
    console8df.to_csv('E:\\weather_grid_data\\kreise_daily_pr\\'+str(kreise)+'.txt',index=False)
    ix=ix+1
    
    
#%% Making the combined economic and climate dataframe

klima_in=pd.read_csv('D:\\KLIMADF_Novv1.csv')
#klima_in.to_csv('D:\\KLIMADF_Novv1.csv',index=False)
econ_in=pd.read_csv('D:\\ECONDF_Novv1.csv')

'''
TMK and for TAS - seasonal and unseasonal -- mean, q3, std, 
TMK and for TAS - hotdays99
ECON vars are percap_gdp, primary, manu, services, with the log_diff
Total columns = 22
'''

full_df=pd.merge(econ_in,klima_in,on=['Kreise_Code','year'],how='inner')

'''
for precipitation, I want the full sum annual precipitation
I want count of days of extremely high precipitation for months
I want the count of very dry days
With precipitation it is necessary to look at dry and wet phenomena
>>>> hotdays99 was developed by taking the distribution for each month over 25 
>>>> years and then counting the number of extreme days for each year relative  
>>>> to modern observations

Annual measures needed for precipitation:
    1. total annual precipitation
    2. seasonally adjusted standard deviation
    3. non-seasonally adjusted standard deviation
    4. number of extreme dry days and extreme wet days
'''
###making the percentile dataframe###
flag='tas'
flag_path='D:\\weather_grid_data\\kreise_grid_daily_tas\\'
files_list=os.listdir(flag_path)
def findpercentile(indf,flag,kreise):
    flaghead=flag+'_2020'
    indf['u99'+flaghead]=0
    indf['l01'+flaghead]=0
    indf['hotdays99'+flaghead]=0
    indf['colddays01'+flaghead]=0
    indf['u95'+flaghead]=0
    indf['l05'+flaghead]=0
    indf['hotdays95'+flaghead]=0
    indf['colddays05'+flaghead]=0
    for month in range(12):
        month=month+1
        indf['month_d']=0
        indf['month_d']=(indf['month']==month).astype(int)
        distbn=indf[indf['month']==month][['MESS_DATUM',flag]]
        distbn=distbn[flag]
        distbn=sorted(distbn)
        indf['u99'+flaghead]=indf['u99'+flaghead]+(indf['month_d']*distbn[int(0.99*len(distbn))])
        indf['l01'+flaghead]=indf['l01'+flaghead]+(indf['month_d']*distbn[int(0.01*len(distbn))])
        indf['hotdays99'+flaghead]=indf['hotdays99'+flaghead]+(indf['month_d']*(indf[flag]>=indf['u99'+flaghead]).astype(int))
        indf['colddays01'+flaghead]=indf['colddays01'+flaghead]+(indf['month_d']*(indf[flag]<=indf['l01'+flaghead]).astype(int))
        indf['u95'+flaghead]=indf['u95'+flaghead]+(indf['month_d']*distbn[int(0.95*len(distbn))])
        indf['l05'+flaghead]=indf['l05'+flaghead]+(indf['month_d']*distbn[int(0.05*len(distbn))])
        indf['hotdays95'+flaghead]=indf['hotdays95'+flaghead]+(indf['month_d']*(indf[flag]>=indf['u95'+flaghead]).astype(int))
        indf['colddays05'+flaghead]=indf['colddays05'+flaghead]+(indf['month_d']*(indf[flag]<=indf['l05'+flaghead]).astype(int))
    indf=indf.drop('month_d',axis=1)
    return indf

var_df3=pd.DataFrame()
for file in files_list:
    df=pd.read_csv(flag_path+file)
    #df['MESS_DATUM']=df['MESS_DATUM'].apply(lambda x: int(x.replace('-','')))
    df['yyyymm']=(df['MESS_DATUM']/100).astype(int)
    df['month']=(df['yyyymm']%100).astype(int)
    df['year']=(df['yyyymm']/100).astype(int)
    df=findpercentile(df, flag,kreise=int(file.split('.')[0]))
    #df.to_csv(flag_path+file)
    flaghead=flag+'_2020'
    daysdf=df[['year',flag,'hotdays99'+flaghead,'hotdays95'+flaghead,'colddays01'+flaghead,'colddays05'+flaghead]]
    daysdf=daysdf.groupby(by='year').sum().reset_index()
    daysdf['Kreise_Code']=int(file.split('.')[0])
    var_df3=pd.concat([var_df3,daysdf],axis=0)

    '''
    for file in files_list:
        infile=pd.read_csv(prec_path+file)   
    stdfile=infile[[flag,'year']].groupby(by='year').std().reset_index()
    stdfile=stdfile.rename(columns={flag:flag+'_std_nsna'})
    stdfile['Kreise_Code']=int(file.split('.')[0])
    stddf=infile[[flag,'yyyymm']].groupby(by='yyyymm').std().reset_index()
    stddf['year']=(stddf['yyyymm']/100).astype(int)
    stddf=stddf[['year',flag]].groupby(by='year').mean().reset_index()
    stddf=stddf.rename(columns={flag:flag+'_std_sna'})
    stddf['Kreise_Code']=int(file.split('.')[0])
    temp=pd.merge(stdfile,stddf,on=['Kreise_Code','year'],how='inner')
    temp=pd.merge(temp,daysdf,on=['Kreise_Code','year'],how='inner')
    '''
    
    
#%% Making seasonal measures : Winter{Dec,Jan,Feb}, Spring{Mar,Apr,May} 
#                              Summer{Jun,Jul,Aug}, Autumn{Sep,Oct,Nov}
'''
make a column of yyyy-season
'''

def findpercentile(indf,flag,feature='tas',annual=False,save=False):
    feature_dict={'tas':['hot','cold'],'prec':['wet','dry']}
    flaghead=flag+'_2020'
    indf=indf[[flag,'MESS_DATUM','quarter','year']]
    indf['u99'+flaghead]=0
    indf['l01'+flaghead]=0
    indf[feature_dict[feature][0]+'99'+flaghead+'_qtr']=0
    indf[feature_dict[feature][1]+'01'+flaghead+'_qtr']=0
    indf['u95'+flaghead]=0
    indf['l05'+flaghead]=0
    indf[feature_dict[feature][0]+'95'+flaghead+'_qtr']=0
    indf[feature_dict[feature][1]+'05'+flaghead+'_qtr']=0
    for quarter in range(4):
        quarter=quarter+1
        indf['quarter_d']=0
        indf['quarter_d']=(indf['quarter']==quarter).astype(int)
        distbn=indf[indf['quarter']==quarter][['MESS_DATUM',flag]]
        distbn=distbn[flag]
        distbn=sorted(distbn)
        indf['u99'+flaghead]=indf['u99'+flaghead]+(indf['quarter_d']*distbn[int(0.99*len(distbn))])
        indf['l01'+flaghead]=indf['l01'+flaghead]+(indf['quarter_d']*distbn[int(0.01*len(distbn))])
        indf[feature_dict[feature][0]+'99'+flaghead+'_qtr']=indf[feature_dict[feature][0]+'99'+flaghead+'_qtr']+(indf['quarter_d']*(indf[flag]>=indf['u99'+flaghead]).astype(int))
        indf[feature_dict[feature][1]+'01'+flaghead+'_qtr']=indf[feature_dict[feature][1]+'01'+flaghead+'_qtr']+(indf['quarter_d']*(indf[flag]<=indf['l01'+flaghead]).astype(int))
        indf['u95'+flaghead]=indf['u95'+flaghead]+(indf['quarter_d']*distbn[int(0.95*len(distbn))])
        indf['l05'+flaghead]=indf['l05'+flaghead]+(indf['quarter_d']*distbn[int(0.05*len(distbn))])
        indf[feature_dict[feature][0]+'95'+flaghead+'_qtr']=indf[feature_dict[feature][0]+'95'+flaghead+'_qtr']+(indf['quarter_d']*(indf[flag]>=indf['u95'+flaghead]).astype(int))
        indf[feature_dict[feature][1]+'05'+flaghead+'_qtr']=indf[feature_dict[feature][1]+'05'+flaghead+'_qtr']+(indf['quarter_d']*(indf[flag]<=indf['l05'+flaghead]).astype(int))
    indf=indf.drop('quarter_d',axis=1)
    if annual==False:
        return indf
    else:
        andf=indf[['year',feature_dict[feature][0]+'99'+flaghead+'_qtr',
                   feature_dict[feature][0]+'95'+flaghead+'_qtr',
                   feature_dict[feature][1]+'01'+flaghead+'_qtr',
                   feature_dict[feature][1]+'05'+flaghead+'_qtr']]
        andf=andf.groupby(by='year').sum().reset_index()
        return andf


flag='tas'
flagpath='D:\\weather_grid_data\\kreise_grid_daily_tas\\'
files_list=os.listdir(flagpath)
season_dict={1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4,12:1}
quarter_dict={1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
season_temp=pd.DataFrame()
season_df=pd.DataFrame()
for file in files_list:
    kreise_code=int(file.split('.')[0])
    indf=pd.read_csv(flagpath+file)
    indf['quarter']=indf['month'].apply(lambda x:quart_dict[x])
    indf['yyyyq']=indf['year']*10+indf['quarter']
    indf['year']=(indf['yyyyq']/10).astype(int)
    '''
    season_temp1=indf[['yyyyq',flag]].groupby('yyyyq').std().reset_index()
    season_temp1['year']=(season_temp1['yyyyq']/10).astype(int)
    season_temp1=season_temp1.groupby(by='year').mean().reset_index()
    season_temp1=season_temp1.rename(columns={flag:flag+'std_sea'})
    season_temp2=indf[['yyyyq',flag]].groupby('yyyyq').mean().reset_index()
    season_temp2['year']=(season_temp2['yyyyq']/10).astype(int)
    season_temp2=season_temp2.groupby(by='year').mean().reset_index()
    season_temp2=season_temp2.rename(columns={flag:flag+'mean_sea'})
    
    temp=pd.merge(season_temp2[['year',flag+'mean_sea']],season_temp1[['year',flag+'std_sea']],on='year',how='inner')
    '''
    percdf=findpercentile(indf,flag=flag,feature=flag,annual=True)
    temp=pd.merge(season_temp1,percdf,on='year',how='inner')
    temp['Kreise_Code']=kreise_code
    
    season_df=pd.concat([season_df,temp],axis=0)

feat='std'
quarter_df2=pd.DataFrame()
for file in files_list:
    kreise_code=int(file.split('.')[0])
    indf=pd.read_csv(flagpath+file)
    indf['quarter']=indf['month'].apply(lambda x:quarter_dict[x])
    indf['yyyyq']=indf['year']*10+indf['quarter']
    indf=indf[['yyyyq',flag]].groupby(by='yyyyq').std().reset_index()
    indf['quarter']=(indf['yyyyq']%10).astype(int)
    indf['quarter1_d']=(indf[['quarter']]==1).astype(int)
    indf['quarter2_d']=(indf[['quarter']]==2).astype(int)
    indf['quarter3_d']=(indf[['quarter']]==3).astype(int)
    indf['quarter4_d']=(indf[['quarter']]==4).astype(int)
    indf['quarter1_'+flag+'_'+feat]=indf[flag]*indf['quarter1_d']
    indf['quarter2_'+flag+'_'+feat]=indf[flag]*indf['quarter2_d']
    indf['quarter3_'+flag+'_'+feat]=indf[flag]*indf['quarter3_d']
    indf['quarter4_'+flag+'_'+feat]=indf[flag]*indf['quarter4_d']
    indf['year']=(indf['yyyyq']/10).astype(int)
    indf=indf[['year','quarter1_'+flag+'_'+feat,'quarter2_'+flag+'_'+feat,'quarter3_'+flag+'_'+feat,'quarter4_'+flag+'_'+feat]].groupby('year').sum().reset_index()
    indf['Kreise_Code']=int(file.split('.')[0])
    quarter_df2=pd.concat([quarter_df2,indf],axis=0)

feat='std'
cols=['quarter1_'+flag+'_'+feat,'quarter2_'+flag+'_'+feat,'quarter3_'+flag+'_'+feat,'quarter4_'+flag+'_'+feat]

    
#%% making the seasonal plots %%#

indf=pd.read_csv('D://weather_grid_data//model_NovV3.csv')
indf=indf[['year','Kreise_Code', 'winter_tas', 'spring_tas', 'summer_tas',
'autumn_tas']]
import matplotlib.pyplot as plt

'''
take the first difference for each county, then for each year [for each season],
count the number of counties that have increased tas or decreased tas.
on the y2 axis, overlay the guitar plot of the tas 
'''
kreise_list=list(set(indf['Kreise_Code']))
diff_sea=pd.DataFrame()
for kreise in kreise_list:
    temp=indf[indf['Kreise_Code']==kreise].drop('Kreise_Code',axis=1)
    temp=temp.set_index('year')
    temp=temp.diff(1).reset_index()
    temp['Kreise_Code']=int(kreise)
    diff_sea=pd.concat([diff_sea,temp],axis=0)




season_plot=pd.DataFrame() #cols=tas_season,up_season,down_season for 4 seasons
for year in range(27):
    year=1995+year
    temp=diff_sea[diff_sea['year']==year]
    temp_diff=temp[[i +'_tas.d1' for i in ['winter','spring','summer','autumn']]]
    temp_diff=(temp_diff>=0).astype(int)
    season_df=pd.DataFrame()
    for season in ['winter','spring','summer','autumn']:
        rise_up=temp_diff[season+'_tas.d1'].sum()
        season_df.loc[season,'rise']=rise_up
        fall=400-rise_up
        season_df.loc[season,'fall']=fall
        season_df.loc[season,'tas']=temp[season+'_tas'].mean()
    season_df=season_df.reset_index()
    season_df['year']=year
    season_plot=pd.concat([season_plot,season_df],axis=0)
    
season_plot=pd.read_csv('D:\\weather_grid_data\\seasonplot.csv')
season_plot['fall']=season_plot['fall']*-1
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
for season in ['winter','spring','summer','autumn']:
    plotdf=season_plot[season_plot['season']==season]
    #season_plot['fall']=season_plot['fall']*-1
    fig,ax1=plt.subplots(figsize=(16,9))
    ax1.bar(x=plotdf['year'],height=plotdf['rise'])
    ax1.bar(x=plotdf['year'],height=plotdf['fall'])
    ax2=ax1.twinx()
    ax2.scatter(x=plotdf['year'],y=plotdf['tas'],color='red',marker='x')
    plt.title('Temperature Observations of German Counties in '+season)
    ax1.set_ylabel('Count of Counties')
    ax2.set_ylabel('Mean Annual Temperature in '+season)
    plt.show()
    
#%% descriptive plotting

indf=pd.read_csv('D:\\weather_grid_data\\model_NovV3.csv')
annual_plot=indf[['year','tas_nsna','winter_tas','spring_tas','summer_tas','autumn_tas']].groupby('year').mean()
annual_plot=annual_plot-annual_plot.mean(axis=0)

#%% making seasonal dataframes %%#
'''
take monthly means for each county for each year
assign the month to the season
assign the county to the bundeslan
four dataframes, one for each season, with monthly average temperature
'''

### FOR GETTING MEAN TEMPERATURE AND TOTAL PRECIPITATION FOR EACH MONTH IN EVERY SEASON
flag='prec'
flag_path='D://weather_grid_data//kreise_daily_pr//'
summerdf,winterdf,springdf,autumndf=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
season_dict={1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4,12:1}
for file in os.listdir(flag_path):
    code=int(file.split('.')[0])
    indf=pd.read_csv(flag_path+file)
    indf=indf[['Kreise_Code','yyyymm',flag]]
    indf['month']=(indf['yyyymm']%100).astype(int)
    indf['season']=indf['month'].apply(lambda x:season_dict[x])
    indf_smr=indf[indf['season']==3]
    indf_wnr=indf[indf['season']==1]
    indf_srn=indf[indf['season']==2]
    indf_aun=indf[indf['season']==4]
    indf_smr=indf_smr[['yyyymm',flag]].groupby(by='yyyymm').sum().reset_index()
    indf_smr['Kreise_Code']=code
    summerdf=pd.concat([summerdf,indf_smr],axis=0)
    indf_wnr=indf_wnr[['yyyymm',flag]].groupby(by='yyyymm').sum().reset_index()
    indf_wnr['Kreise_Code']=code
    winterdf=pd.concat([winterdf,indf_wnr],axis=0)
    indf_aun=indf_aun[['yyyymm',flag]].groupby(by='yyyymm').sum().reset_index()
    indf_aun['Kreise_Code']=code
    autumndf=pd.concat([autumndf,indf_aun],axis=0)
    indf_srn=indf_srn[['yyyymm',flag]].groupby(by='yyyymm').sum().reset_index()
    indf_srn['Kreise_Code']=code
    springdf=pd.concat([springdf,indf_srn],axis=0)


### FOR GETTING SEASONAL STANDARD-DEVIATION FOR

flag='prec'
flag_path='D://weather_grid_data//kreise_daily_pr//'
summerdf,winterdf,springdf,autumndf=pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
season_dict={1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4,12:1}
for file in os.listdir(flag_path):
    code=int(file.split('.')[0])
    indf=pd.read_csv(flag_path+file)
    indf=indf[['Kreise_Code','year','yyyymm',flag]]
    indf['month']=(indf['yyyymm']%100).astype(int)
    indf['season']=indf['month'].apply(lambda x:season_dict[x])
    indf['yyyys']=indf['year']*10+indf['season']
    indf_smr=indf[indf['season']==3]
    indf_wnr=indf[indf['season']==1]
    indf_srn=indf[indf['season']==2]
    indf_aun=indf[indf['season']==4]
    indf_smr=indf_smr[['year',flag]].groupby(by='year').std().reset_index()
    indf_smr['Kreise_Code']=code
    indf_smr=indf_smr.rename(columns={flag:'summer_'+flag+'_std'})
    summerdf=pd.concat([summerdf,indf_smr],axis=0)
    indf_wnr=indf_wnr[['year',flag]].groupby(by='year').std().reset_index()
    indf_wnr['Kreise_Code']=code
    indf_wnr=indf_wnr.rename(columns={flag:'winter_'+flag+'_std'})
    winterdf=pd.concat([winterdf,indf_wnr],axis=0)
    indf_aun=indf_aun[['year',flag]].groupby(by='year').std().reset_index()
    indf_aun['Kreise_Code']=code
    indf_aun=indf_aun.rename(columns={flag:'autumn_'+flag+'_std'})
    autumndf=pd.concat([autumndf,indf_aun],axis=0)
    indf_srn=indf_srn[['year',flag]].groupby(by='year').std().reset_index()
    indf_srn['Kreise_Code']=code
    indf_srn=indf_srn.rename(columns={flag:'spring_'+flag+'_std'})
    springdf=pd.concat([springdf,indf_srn],axis=0)

import geopandas as gpd
kreise_file=gpd.read_file('D:\\archive\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_file=kreise_file[['Kreise_Code','geometry']]
kreise_list=list(kreise_file['Kreise_Code'])  
season_files=['springdfprec.csv','autumndfprec.csv','winterdfprec.csv','summerdfprec.csv']
flag='prec'
fig,ax=plt.subplots(2,2,figsize=(32,18))
ix=0
for file in season_files:
    season=file.split('d')[0]
    indf=pd.read_csv('D://weather_grid_data//'+file)
    indf['year']=(indf['yyyymm']/100).astype(int)
    anmap=indf[['Kreise_Code','year',flag]].groupby(by=['Kreise_Code','year']).sum()
    anmap=anmap.reset_index()
    anmap[flag+'_an']=anmap[['year',flag]].groupby(by='year').transform('mean')
    anmap[flag+'_kr']=anmap[['Kreise_Code',flag]].groupby(by='Kreise_Code').transform('mean')
    anmap[flag+'MEAN']=anmap[flag].mean()
    anmap[flag+'.dm']=anmap[flag]-anmap[flag+'_an']-anmap[flag+'_kr']+anmap[flag+'MEAN']
    anmap[flag+'.dm.d1']=anmap[['Kreise_Code',flag+'.dm']].groupby(by='Kreise_Code').transform('diff')
    anmap=anmap.dropna()
    anmap['yoy_rise']=(anmap[flag+'.dm.d1']>0).astype(int)
    ants=anmap[['year',flag]].groupby(by='year').mean().reset_index()
    anct=anmap[['year','yoy_rise']].groupby(by='year').sum().reset_index()
    anct['yoy_fall']=(400-anct['yoy_rise'])*-1
    ax1=ax[int(ix/2)][int(ix%2)]
    ax2=ax1.twinx()
    ax1.bar(x=anct['year'],height=anct['yoy_rise'],color='red',alpha=0.6,label='yoy_rise')
    ax1.bar(x=anct['year'],height=anct['yoy_fall'],color='green',alpha=0.6,label='yoy_fall')
    ax1.legend()
    ax2.plot(ants.set_index('year')[flag],color='black',linewidth=1.3)
    ax1.set_title('Year-on-Year (2wE adj) Precipitation Dynamics in '+season.capitalize(),fontsize=25)
    ax1.set_ylabel('Number of Counties',fontsize=18)
    ax2.set_ylabel('Total Precipitation ',fontsize=18)
    ix=ix+1
plt.show()
    
season_files=['springdfprec.csv','autumndfprec.csv','winterdfprec.csv','summerdfprec.csv']
flag='prec'
fig,ax=plt.subplots(2,2,figsize=(32,18))
ix=0
for file in season_files:
    season=file.split('d')[0]
    indf=pd.read_csv('D://weather_grid_data//'+file)
    indf['year']=(indf['yyyymm']/100).astype(int)
    anmap=indf[['Kreise_Code','year',flag]].groupby(by=['Kreise_Code','year']).sum()
    #anmap=anmap.reset_index()
    anmap[flag+'.d1']=anmap.groupby(by='Kreise_Code').transform('diff')
    anmap=anmap.dropna()
    anmap=anmap.reset_index()
    anmap['yoy_rise']=(anmap[flag+'.d1']>0).astype(int)
    ants=anmap[['year',flag]].groupby(by='year').mean().reset_index()
    anct=anmap[['year','yoy_rise']].groupby(by='year').sum().reset_index()
    anct['yoy_fall']=400-anct['yoy_rise']
    ants=pd.merge(anct[['year','yoy_rise','yoy_fall']],ants[['year',flag]],on='year',how='inner')
    ants['yoy_fall']=ants['yoy_fall']*-1
    ax1=ax[int(ix/2)][int(ix%2)]
    ax2=ax1.twinx()
    ax1.bar(x=ants['year'],height=ants['yoy_rise'],color='red',alpha=0.6,label='yoy_rise')
    ax1.bar(x=ants['year'],height=ants['yoy_fall'],color='green',alpha=0.6,label='yoy_fall')
    ax1.legend()
    ax2.plot(ants.set_index('year')[flag],color='black',linewidth=1.3)
    ax1.set_title('Year-on-Year Precipitation Dynamics in '+season.capitalize(),fontsize=30)
    ax1.set_ylabel('Number of Counties',fontsize=18)
    ax2.set_ylabel('Total Precipitation',fontsize=18)
    ix=ix+1
#.supylabel('Number of Counties showing y-o-y Rise (+) or Fall (-)',fontsize=22)
#ax2.supylabel('Mean Temperature for the Season each Year',fontsize=22)
#plt.legend()
plt.show()

for file in season_files:
    season=file.split('d')[0]
    fig_map,ax_map=plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    indf=pd.read_csv('D://weather_grid_data//seasonal_df//'+file)
    indf['year']=(indf['yyyymm']/100).astype(int)
    anmap=indf[['Kreise_Code','year',flag]].groupby(by=['Kreise_Code','year']).sum().reset_index()
    max_flag,min_flag=anmap[flag].max(),anmap[flag].min()
    #anmap=pd.merge(kreise_file,anmap,on=['Kreise_Code'],how='inner')
    anmap1=anmap[anmap['year']>=1995][['year','Kreise_Code',flag]]
    anmap1=anmap1[anmap1['year']<=2000]
    anmap1=anmap1.groupby(by='Kreise_Code').mean().reset_index()
    anmap1=pd.merge(kreise_file,anmap1,on='Kreise_Code',how='inner')
    anmap1.plot(column=anmap1[flag],ax=ax_map[0],cmap=season+'_r',legend=True,vmax=max_flag,vmin=min_flag)
    ax_map[0].set_title('1995-2000',fontsize=20)
    ax_map[0].axis('off')
    anmap2=anmap[anmap['year']>=2015][['year','Kreise_Code',flag]]
    anmap2=anmap2[anmap2['year']<=2020]
    anmap2=anmap2.groupby(by='Kreise_Code').mean().reset_index()
    anmap2=pd.merge(kreise_file,anmap2,on='Kreise_Code',how='inner')
    anmap2.plot(column=anmap2[flag],ax=ax_map[1],cmap=season+'_r',legend=True,vmax=max_flag,vmin=min_flag)
    ax_map[1].set_title('2015-2020',fontsize=20)
    ax_map[1].axis('off')
    fig_map.suptitle('Mean (Total) Annual Precipitation (mm) in '+season.capitalize(),fontsize=25)
    plt.show()

season_files=['springdf.csv','summerdf.csv','autumndf.csv','winterdf.csv']
flag='tas'
for file in season_files:
    season=file.split('d')[0]
    #fig_map,ax_map=plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    indf=pd.read_csv('D://weather_grid_data//seasonal_df//'+file)
    indf['year']=(indf['yyyymm']/100).astype(int)
    anmap=indf[['Kreise_Code','year',flag]].groupby(by=['Kreise_Code','year']).sum().reset_index()
    anmap1=anmap[anmap['year']>=1995][['year','Kreise_Code',flag]]
    anmap1=anmap1[anmap1['year']<=2000]
    anmap1=anmap1[['Kreise_Code',flag]].groupby(by='Kreise_Code').mean().reset_index()
    anmap1=anmap1.rename(columns={flag:flag+'_I'})
    anmap2=anmap[anmap['year']>=2015][['year','Kreise_Code',flag]]
    anmap2=anmap2[anmap2['year']<=2020]
    anmap2=anmap2[['Kreise_Code',flag]].groupby(by='Kreise_Code').mean().reset_index()
    anmap2=anmap2.rename(columns={flag:flag+'_II'})
    anmap=pd.merge(anmap1,anmap2,on=['Kreise_Code'])
    anmap.to_csv('D://weather_grid_data//seasonal_df//transition_'+flag+'_'+file,index=False)

fig,ax=plt.subplots(2,2,figsize=(20,20))
transition_files=['transition_'+flag+'_'+file for file in season_files]
ix=0
colors={'Spring':'fuchsia','Autumn':'darkorange','Winter':'cyan','Summer':'lawngreen'}
for transition_file in transition_files:
    season=transition_file.split('_')[2]
    season=season.split('d')[0].capitalize()
    indf=pd.read_csv('D://weather_grid_data//seasonal_df//'+transition_file)
    indf['Δ.(II-I)']=indf[flag+'_II']-indf[flag+'_I']
    seaborn.histplot(indf['Δ.(II-I)'],ax=ax[int(ix/2)][int(ix%2)],color=colors[season],kde=True,line_kws={'linewidth':5,'color':'black'})
    ax[int(ix/2)][int(ix%2)].set_title('Δ.(II-I) for Precipitation in '+season,fontsize=25)
    ix=ix+1
plt.show()
    
    
fig_map,ax_map=plt.subplots(nrows=1,ncols=2,figsize=(16,9))
ax_map[0].set_title('1995')
ax_map[1].set_title('2020')
max_flag=anmap['tas'].max()
anmap1=anmap[anmap['year']<=2004][['Kreise_Code',flag]].groupby(by='Kreise_Code').mean().reset_index()
anmap1=pd.merge(kreise_file,anmap1,on='Kreise_Code',how='inner')
anmap1.plot(column=anmap1[flag],ax=ax_map[0],cmap=season+'_r',legend=True,vmax=max_flag)
anmap2=anmap[anmap['year']>2004][['Kreise_Code',flag]].groupby(by='Kreise_Code').mean().reset_index()
anmap2=pd.merge(kreise_file,anmap2,on='Kreise_Code',how='inner')
anmap2.plot(column=anmap2[flag],ax=ax_map[1],cmap=season+'_r',legend=True,vmax=max_flag)
fig_map.suptitle('Mean Temperature in '+season)
plt.show()

#%%% making time series plot

indf=pd.read_csv('D://weather_grid_data/model_NovV3.csv')
indf['anmin_tas']=indf[['year','tas_nsna']].groupby(by='year').transform('min')
indf['anmean_tas']=indf[['year','tas_nsna']].groupby(by='year').transform('mean')
indf['anmax_tas']=indf[['year','tas_nsna']].groupby(by='year').transform('max')
indf['wnmin_tas']=indf[['year','winter_tas']].groupby(by='year').transform('min')
indf['wnmean_tas']=indf[['year','winter_tas']].groupby(by='year').transform('mean')
indf['wnmax_tas']=indf[['year','winter_tas']].groupby(by='year').transform('max')
indf['smmin_tas']=indf[['year','summer_tas']].groupby(by='year').transform('min')
indf['smmean_tas']=indf[['year','summer_tas']].groupby(by='year').transform('mean')
indf['smmax_tas']=indf[['year','summer_tas']].groupby(by='year').transform('max')
indf['pgmin_tas']=indf[['year','spring_tas']].groupby(by='year').transform('min')
indf['pgmean_tas']=indf[['year','spring_tas']].groupby(by='year').transform('mean')
indf['pgmax_tas']=indf[['year','spring_tas']].groupby(by='year').transform('max')
indf['aumin_tas']=indf[['year','autumn_tas']].groupby(by='year').transform('min')
indf['aumean_tas']=indf[['year','autumn_tas']].groupby(by='year').transform('mean')
indf['aumax_tas']=indf[['year','autumn_tas']].groupby(by='year').transform('max')

time_plot1=indf[['year','anmin_tas','anmax_tas','anmean_tas',
                        'wnmin_tas','wnmean_tas','wnmax_tas',
                        'smmin_tas','smmean_tas','smmax_tas',
                        'pgmin_tas','pgmean_tas','pgmax_tas',
                        'aumin_tas','aumean_tas','aumax_tas']].drop_duplicates(keep='first')
time_plot1=time_plot1.set_index('year')
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')

fig,ax=plt.subplots(2,2,figsize=(32,18))
ix=0
deg=5
ses={'wn':'Winter','pg':'Spring','sm':'Summer','au':'Autumn'}
for season in ses:
    temp=time_plot1[[season+'min_tas',season+'mean_tas',season+'max_tas']]
    temp=temp.reset_index()
    plot_temp=pd.DataFrame()
    plot_temp['year']=temp['year']
    polyfit_min=np.polyfit(plot_temp['year'],temp[season+'min_tas'],deg=deg)
    polyfit_mean=np.polyfit(plot_temp['year'],temp[season+'mean_tas'],deg=deg)
    polyfit_max=np.polyfit(plot_temp['year'],temp[season+'max_tas'],deg=deg)
    plot_temp[season+'min_tas']=np.polyval(polyfit_min,plot_temp['year'])
    plot_temp[season+'mean_tas']=np.polyval(polyfit_mean,plot_temp['year'])
    plot_temp[season+'max_tas']=np.polyval(polyfit_max,plot_temp['year'])
    plot_temp=plot_temp.set_index('year')
    plot_temp.plot(ax=ax[int(ix/2)][int(ix%2)])
    ax[int(ix/2)][int(ix%2)].set_title('Time Series of '+ses[season]+' Temperatures',fontsize=30)
    ix=ix+1
plt.show()

#%% making the plot for cross-sectional ols beta coefficient with indication if it is significant

from statsmodels.api import OLS
novv3=pd.read_csv('D://weather_grid_data//model_NovV3.csv')
years=list(set(novv3['year']))
regdf=novv3[['year','tas_nsna','winter_tas','spring_tas','summer_tas','autumn_tas','ln_diff(1)']]
exog=['tas_nsna','winter_tas','spring_tas','summer_tas','autumn_tas']
endog='ln_diff(1)'
params_df=pd.DataFrame(index=years)
for year in years:
    test=regdf[regdf['year']==year]
    test['constant']=1
    model=OLS(test[endog],test[exog+['constant']])
    model=model.fit()
    params_df.loc[year,'β-tas_nsna']=model.params['tas_nsna']
    params_df.loc[year,'p-val tas_nsna']=model.pvalues['tas_nsna']
    params_df.loc[year,'β-win_tas']=model.params['winter_tas']
    params_df.loc[year,'p-val win_tas']=model.pvalues['winter_tas']
    params_df.loc[year,'β-sum_tas']=model.params['summer_tas']
    params_df.loc[year,'p-val sum_tas']=model.pvalues['summer_tas']
    params_df.loc[year,'β-aun_tas']=model.params['autumn_tas']
    params_df.loc[year,'p-val aun_tas']=model.pvalues['autumn_tas']
    params_df.loc[year,'β-tas_C']=model.params['constant']
    params_df.loc[year,'p-val C']=model.pvalues['constant']
    
params_df=params_df.drop(['β-tas_C','p-val C'],axis=1)

for item in [i for i in params_df.columns if 'p-val' in i]:
    params_df[item]=(params_df[item]<=0.1).astype(int)
    tt=item.replace('p-val ','β-')
    params_df[item]=params_df[item]*params_df[tt]


fig,ax=plt.subplots(figsize=(13,6))
ax.plot(params_df[[i for i in params_df.columns if 'β-' in i]],label=[i for i in params_df.columns if 'β-' in i])
plt.title('β Coefficient for Cross-sectional Regression ',fontsize=25)
for item in [i for i in params_df.columns if 'p-val' in i]:
    pval=params_df[params_df[item]!=0][item].reset_index()
    pval=pval.rename(columns={'index':'year'})
    pval['year']=pval['year'].astype(int)
    ax.scatter(pval['year'],pval[item],color='black',marker='x',label='p<0.1')
plt.legend([i for i in params_df.columns if 'β-' in i]+['p<0.1'])
plt.show()

#%% make a file for each year, with all counties daily observations

years=[1990+i for i in range(31)]
kreises=os.listdir('D://weather_grid_data//kreise_grid_daily_tas//')
for year in years:
    yeardf=pd.DataFrame()
    for kreise in kreises:
        infile=pd.read_csv('D://weather_grid_data//kreise_grid_daily_tas//'+kreise)
        infile=infile[['MESS_DATUM','year','yyyymm','tas','Kreise_Code']]
        infile=infile[infile['year']==year]
        yeardf=pd.concat([yeardf,infile],axis=0)
    yeardf.to_csv('D://weather_grid_data//year_df//year_daily//'+str(year)+'.csv',index=False)
    
for year in years:
    yeardf=pd.read_csv('D://weather_grid_data//year_df//year_daily//'+str(year)+'.csv')
    plt.hist(yeardf['tas'],density=True)
    plt.title(str(year))
    plt.show()
    
novv3=pd.read_csv('D://weather_grid_data//model_df//model_NovV3.csv')
novv3['tasmean_sea.d1']=novv3[['Kreise_Code','tasmean_sea']].groupby(by=['Kreise_Code']).transform('diff')
novv3['tas_nsna.d1']=novv3[['Kreise_Code','year','tas_nsna']].groupby(by=['Kreise_Code']).transform('diff')
novv3['winter_tas.d1']=novv3[['Kreise_Code','year','winter_tas']].groupby(by=['Kreise_Code']).transform('diff')
novv3['summer_tas.d1']=novv3[['Kreise_Code','year','summer_tas']].groupby(by=['Kreise_Code']).transform('diff')
novv3['spring_tas.d1']=novv3[['Kreise_Code','year','spring_tas']].groupby(by=['Kreise_Code']).transform('diff')
novv3['autumn_tas.d1']=novv3[['Kreise_Code','year','autumn_tas']].groupby(by=['Kreise_Code']).transform('diff')

df=pd.DataFrame()
df=novv3[['year','Kreise_Code']]
for col in ['tasmean_sea','tas_nsna','winter_tas','summer_tas','spring_tas','autumn_tas']:
    temp=novv3[['Kreise_Code','year',col]]
    temp[col+'.d1']=temp[['Kreise_Code',col]].groupby(by='Kreise_Code')[col].transform('diff')
    df=pd.merge(df,temp,on=['year','Kreise_Code'],how='outer')
    
novv3=pd.merge(novv3,df[[i for i in df.columns if '.d1' in i]+['Kreise_Code','year']],on=['Kreise_Code','year'],how='inner')

inecon=pd.read_csv('D://weather_grid_data/model_df//model_NovV3.csv')
inecon=inecon[['year','Kreise_Code']+['primary_GVA', 'svc_GVA', 'man_GVA',
       'gdp_percap']]

fig,ax=plt.subplots(1,2,figsize=(14,7))
ix=0
for factor in ['primary_GVA','man_GVA']:#'primary_GVA', 'svc_GVA'
    temp=inecon[['year','Kreise_Code',factor]]
    temp=temp.dropna()

    tt=temp[['Kreise_Code','year']].groupby(by='year').count().reset_index()
    max_kreise=tt['Kreise_Code'].max()
    min_year=tt[tt['Kreise_Code']==max_kreise]['year'].min()
    max_year=2020
    
    temp1=temp[temp['year']==min_year]
    temp1[factor]=log(temp1[factor])
    temp2=temp[temp['year']==max_year]
    temp2[factor]=log(temp2[factor])
    
    temp1=temp1.rename(columns={factor:factor+'_'+str(min_year)})
    temp1=temp1.drop('year',axis=1)
    temp2=temp2.rename(columns={factor:factor+'_'+str(max_year)})
    temp2=temp2.drop('year',axis=1)
    temp=pd.merge(temp1,temp2,on='Kreise_Code',how='inner')
    temp['Δ.Ln('+factor+')']=temp[factor+'_'+str(max_year)]-temp[factor+'_'+str(min_year)]
    #temp=temp.drop('year',axis=1)
    temp=pd.merge(kreise_file,temp,on='Kreise_Code',how='inner')
    temp.plot(column=temp['Δ.Ln('+factor+')'],ax=ax[ix],cmap='gist_heat',legend=True)
    ax[ix].set_title('Δ.Ln('+factor+'):'+str(min_year)+' - '+str(max_year),fontsize=20)
    ax[ix].axis('off')
    ix=ix+1
plt.show()
    


def find_season_percentile(df,flag,season='summer',path=''):
    df['year']=(df['yyyys']/10).astype(int)
    flag_dist=df[df['year']<=2021][flag]
    flag_dist=list(flag_dist)
    flag_dist=sorted(flag_dist)
    high95,high99=flag_dist[int(0.95*len(flag_dist))],flag_dist[int(0.99*len(flag_dist))]
    low05,low01=flag_dist[int(0.05*len(flag_dist))],flag_dist[int(0.01*len(flag_dist))]
    df[season+'high95_'+flag]=high95
    df[season+'high99_'+flag]=high99
    df[season+'low05_'+flag]=low05
    df[season+'low01_'+flag]=low01
    df[season+'high95_'+flag+'days']=(df[flag]>=high95).astype(int)
    df[season+'high99_'+flag+'days']=(df[flag]>=high99).astype(int)
    df[season+'low01_'+flag+'days']=(df[flag]<=low01).astype(int)
    df[season+'low05_'+flag+'days']=(df[flag]<=low05).astype(int)
    df['year']=(df['yyyys']/10).astype(int)
    percs=df[['year',season+'high95_'+flag,season+'high99_'+flag,season+'low05_'+flag,season+'low01_'+flag]].drop_duplicates()
    #df.to_csv(path,index=False)
    df=df[['year',season+'high95_'+flag+'days',season+'high99_'+flag+'days',
               season+'low01_'+flag+'days',season+'low05_'+flag+'days']].groupby(by='year').sum()
    df=pd.merge(df,percs,on='year',how='inner')
    return df
#How has winter changed in Germany?
#Mark the number of extreme days in winter.
#Mark the number of high-precipitation days
#Mark the total winter precipitation
#Mark the ice-fall days, that neg_temp*prec

flag='prec'
flag_path='D://weather_grid_data//kreise_daily_pr//'
summerdf=pd.DataFrame()
winterdf,springdf,autumndf=pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
season_dict={1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4,12:1}
for file in files_list:
    code=file
    indf=pd.read_csv(flag_path+file)
    indf=indf[['Kreise_Code','yyyymm','year',flag]]
    indf['month']=(indf['yyyymm']%100).astype(int)
    indf['season']=indf['month'].apply(lambda x:season_dict[x])
    indf['yyyys']=(indf['year']*10+indf['season']).astype(int)
    indf=indf.drop(['yyyymm','year'],axis=1)
    indf=indf[['yyyys','season',flag]]
    indf_smr=indf[indf['season']==3]
    indf_wnr=indf[indf['season']==1]
    indf_spg=indf[indf['season']==2]
    indf_aun=indf[indf['season']==4]
    
    part_sum=indf_smr[['yyyys',flag]].groupby(by='yyyys').sum().reset_index()
    part_sum=part_sum.rename(columns={flag:'summersum_'+flag})
    part_std=indf_smr[['yyyys',flag]].groupby(by='yyyys').std().reset_index()
    part_std=part_std.rename(columns={flag:'summerstd_'+flag})
    #part_perc= find_season_percentile(df=indf_smr,flag=flag,season='summer',path=flag_path+'summer'+file).reset_index()
    indf_smr=pd.merge(part_sum,part_std,on='yyyys',how='inner')
    indf_smr['year']=(indf_smr['yyyys']/10).astype(int)
    indf_smr=indf_smr.drop('yyyys',axis=1)
    #indf_smr=pd.merge(indf_smr,part_perc,on='year',how='inner')
    indf_smr['Kreise_Code']=code
    summerdf=pd.concat([summerdf,indf_smr],axis=0)

    part_sum=indf_wnr[['yyyys',flag]].groupby(by='yyyys').sum().reset_index()
    part_sum=part_sum.rename(columns={flag:'wintersum_'+flag})
    part_std=indf_wnr[['yyyys',flag]].groupby(by='yyyys').std().reset_index()
    part_std=part_std.rename(columns={flag:'winterstd_'+flag})
    #part_perc= find_season_percentile(df=indf_wnr,flag=flag,season='winter',path=flag_path+'winter'+file).reset_index()
    indf_wnr=pd.merge(part_sum,part_std,on='yyyys',how='inner')
    indf_wnr['year']=(indf_wnr['yyyys']/10).astype(int)
    indf_wnr=indf_wnr.drop('yyyys',axis=1)
    #indf_wnr=pd.merge(indf_wnr,part_perc,on='year',how='inner')
    indf_wnr['Kreise_Code']=code
    winterdf=pd.concat([winterdf,indf_wnr],axis=0)
    
    part_sum=indf_aun[['yyyys',flag]].groupby(by='yyyys').sum().reset_index()
    part_sum=part_sum.rename(columns={flag:'autumnsum_'+flag})
    part_std=indf_aun[['yyyys',flag]].groupby(by='yyyys').std().reset_index()
    part_std=part_std.rename(columns={flag:'autumnstd_'+flag})
    #part_perc= find_season_percentile(df=indf_aun,flag=flag,season='autumn',path=flag_path+'autumn'+file).reset_index()
    indf_aun=pd.merge(part_sum,part_std,on='yyyys',how='inner')
    indf_aun['year']=(indf_aun['yyyys']/10).astype(int)
    indf_aun=indf_aun.drop('yyyys',axis=1)
    #indf_aun=pd.merge(indf_aun,part_perc,on='year',how='inner')
    indf_aun['Kreise_Code']=code
    autumndf=pd.concat([autumndf,indf_aun],axis=0)
    
    part_sum=indf_spg[['yyyys',flag]].groupby(by='yyyys').sum().reset_index()
    part_sum=part_sum.rename(columns={flag:'springsum_'+flag})
    part_std=indf_spg[['yyyys',flag]].groupby(by='yyyys').std().reset_index()
    part_std=part_std.rename(columns={flag:'springstd_'+flag})
    #part_perc= find_season_percentile(df=indf_spg,flag=flag,season='spring',path=flag_path+'spring'+file).reset_index()
    indf_spg=pd.merge(part_sum,part_std,on='yyyys',how='inner')
    indf_spg['year']=(indf_spg['yyyys']/10).astype(int)
    indf_spg=indf_spg.drop('yyyys',axis=1)
    #indf_spg=pd.merge(indf_spg,part_perc,on='year',how='inner')
    indf_spg['Kreise_Code']=code
    springdf=pd.concat([springdf,indf_spg],axis=0)
    
    
#%%% MAKING A DATAFRAME OF ALL RELEVANT ANNUAL MEASURES OF CLIMATE VARIABLES

#1. TEMPERATURE DATA

##.percentile data sets
'''
percentile data sets will calculate monthly and seasonally adjusted -01th, 
-05th, 95th and 99th from before 1995 and adjusted till 2020. Then calculate
the number of days each year that fall in the tails for each county.
make a history-reference dataset for all counties with the tail values
this is 400 counties data with the history percentiles and present percentiles

these will also find the highest 90th and lowest 10th temperatures for each 
county in each year
'''

def kreise_Percentile(df,flag,kreise_code,hist):
    suffix=str(int(hist%100))
    outdf_mn=pd.DataFrame(columns=['Kreise_Code','month','cold01_mn.'+suffix,'cold05_mn.'+suffix,'hot95_mn.'+suffix,'hot99_mn.'+suffix])
    hist_df=df[df['year']<hist]
    for month in range(12):
        month=month+1
        flag_dist=list(hist_df[hist_df['month']==month][flag])
        flag_dist,flag_len=sorted(flag_dist),len(flag_dist)
        outdf_mn.loc[month-1,'month']=int(month)
        outdf_mn.loc[month-1,'cold01_mn.'+suffix]=flag_dist[int(0.01*flag_len)]
        outdf_mn.loc[month-1,'cold05_mn.'+suffix]=flag_dist[int(0.05*flag_len)]
        outdf_mn.loc[month-1,'hot95_mn.'+suffix]=flag_dist[int(0.95*flag_len)]
        outdf_mn.loc[month-1,'hot99_mn.'+suffix]=flag_dist[int(0.99*flag_len)]
    outdf_mn['Kreise_Code']=kreise_code
    return outdf_mn

flag='tas'
flag_path='D://weather_grid_data//tasfull//'
files_list=os.listdir(flag_path)
history=1995
months_df=pd.DataFrame()
for file in files_list:
    kreise_code=int(file.split('.')[0])
    infile=pd.read_csv(flag_path+file)
    outdf=kreise_Percentile(infile,flag=flag,kreise_code=kreise_code,hist=history) ##change history to 1995,2001,2021

season_dict={1:1,2:1,3:2,4:2,5:2,6:3,7:3,8:3,9:4,10:4,11:4,12:1}
def kreise_Percentile(df,flag,kreise_code,hist):
    suffix=str(int(hist%100))
    outdf_sea=pd.DataFrame(columns=['Kreise_Code','season','cold01_sea.'+suffix,'cold05_sea.'+suffix,'hot95_sea.'+suffix,'hot99_sea.'+suffix])
    hist_df=df[df['year']<hist]
    for season in range(4):
        season=season+1
        flag_dist=list(hist_df[hist_df['season']==season][flag])
        flag_dist,flag_len=sorted(flag_dist),len(flag_dist)
        outdf_sea.loc[season-1,'season']=int(season)
        outdf_sea.loc[season-1,'cold01_sea.'+suffix]=flag_dist[int(0.01*flag_len)]
        outdf_sea.loc[season-1,'cold05_sea.'+suffix]=flag_dist[int(0.05*flag_len)]
        outdf_sea.loc[season-1,'hot95_sea.'+suffix]=flag_dist[int(0.95*flag_len)]
        outdf_sea.loc[season-1,'hot99_sea.'+suffix]=flag_dist[int(0.99*flag_len)]
    outdf_sea['Kreise_Code']=kreise_code
    return outdf_sea


#files_list=os.listdir(flag_path)

seasons_df=pd.DataFrame()
for file in files_list:
    kreise_code=int(file.split('.')[0])
    infile=pd.read_csv(flag_path+file)
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    outdf=kreise_Percentile(infile,flag=flag,kreise_code=kreise_code,hist=history)
    seasons_df=pd.concat([seasons_df,outdf],axis=0)
    
sea_perc=pd.read_csv('D://weather_grid_data//seasonal_df/seasonal_percentiles_tas.csv')
mon_perc=pd.read_csv('D:/weather_grid_data/seasonal_df/monthly_percentiles_tas.csv')  
count_df=pd.DataFrame()

### find the number of hot days in summer as an indicator of heat-stress events in German counties ###
smr_stress=pd.DataFrame()
files_list=[str(i)+'.txt' for i in appendcounties]
for file in files_list:
    infile=pd.read_csv(flag_path+file)
    kreise_code=int(file.split('.')[0])
    sea_bound=sea_perc[sea_perc['Kreise_Code']==kreise_code]
    smr_bound=sea_bound[sea_bound['season']==3]['hot99_sea.1'].values[0]
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    infile=infile[infile['season']==3]
    infile['hot99_smr.1_'+flag+'.d']=(infile[flag]>=smr_bound).astype(int)
    tempdf=infile[['year','hot99_smr.1_'+flag+'.d']]
    tempdf=tempdf.groupby('year').sum().reset_index()
    tempdf['Kreise_Code']=kreise_code
    smr_stress=pd.concat([smr_stress,tempdf],axis=0)


flag='prec'
flag_path='D://weather_grid_data//kreise_daily_pr//'
files_list=os.listdir(flag_path)

count_df=pd.DataFrame()
for file in files_list:
    infile=pd.read_csv(flag_path+file)
    #infile=infile.drop(['u99tas_2010', 'l01tas_2010', 'hotdays99tas_2010', 'colddays01tas_2010',
    #'u99tas_2020', 'l01tas_2020', 'hotdays99tas_2020', 'colddays01tas_2020'],axis=1)
    kreise_code=int(file.split('.')[0])
    sea_bounds,mn_bounds=sea_perc[sea_perc['Kreise_Code']==kreise_code],mon_perc[mon_perc['Kreise_Code']==kreise_code]
    sea_bounds=sea_bounds.set_index('season')
    mn_bounds=mn_bounds.set_index('month')
    
    infile['cold01_mn.95_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'cold01_mn.95'])
    infile['cold05_mn.95_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'cold05_mn.95'])
    infile['hot95_mn.95_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'hot95_mn.95'])
    infile['hot99_mn.95_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'hot99_mn.95'])
    infile['cold01_mn.1_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'cold01_mn.1'])
    infile['cold05_mn.1_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'cold05_mn.1'])
    infile['hot95_mn.1_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'hot95_mn.1'])
    infile['hot99_mn.1_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'hot99_mn.1'])
    infile['cold01_mn.21_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'cold01_mn.21'])
    infile['cold05_mn.21_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'cold05_mn.21'])
    infile['hot95_mn.21_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'hot95_mn.21'])
    infile['hot99_mn.21_'+flag]=infile['month'].apply(lambda x: mn_bounds.loc[x,'hot99_mn.21'])
    
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    infile['cold01_sea.95_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'cold01_sea.95'])
    infile['cold05_sea.95_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'cold05_sea.95'])
    infile['hot95_sea.95_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'hot95_sea.95'])
    infile['hot99_sea.95_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'hot99_sea.95'])
    infile['cold01_sea.1_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'cold01_sea.1'])
    infile['cold05_sea.1_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'cold05_sea.1'])
    infile['hot95_sea.1_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'hot95_sea.1'])
    infile['hot99_sea.1_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'hot99_sea.1'])
    infile['cold01_sea.21_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'cold01_sea.21'])
    infile['cold05_sea.21_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'cold05_sea.21'])
    infile['hot95_sea.21_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'hot95_sea.21'])
    infile['hot99_sea.21_'+flag]=infile['season'].apply(lambda x: sea_bounds.loc[x,'hot99_sea.21'])
    
    infile['cold01_mn.95_'+flag+'.d']=(infile[flag]<=infile['cold01_mn.95_'+flag]).astype(int)
    infile['cold05_mn.95_'+flag+'.d']=(infile[flag]<=infile['cold05_mn.95_'+flag]).astype(int)
    infile['hot95_mn.95_'+flag+'.d']=(infile[flag]>=infile['hot95_mn.95_'+flag]).astype(int)
    infile['hot99_mn.95_'+flag+'.d']=(infile[flag]>=infile['hot99_mn.95_'+flag]).astype(int)
    infile['cold01_mn.1_'+flag+'.d']=(infile[flag]<=infile['cold01_mn.1_'+flag]).astype(int)
    infile['cold05_mn.1_'+flag+'.d']=(infile[flag]<=infile['cold05_mn.1_'+flag]).astype(int)
    infile['hot95_mn.1_'+flag+'.d']=(infile[flag]>=infile['hot95_mn.1_'+flag]).astype(int)
    infile['hot99_mn.1_'+flag+'.d']=(infile[flag]>=infile['hot99_mn.1_'+flag]).astype(int)
    infile['cold01_mn.21_'+flag+'.d']=(infile[flag]<=infile['cold01_mn.21_'+flag]).astype(int)
    infile['cold05_mn.21_'+flag+'.d']=(infile[flag]<=infile['cold05_mn.21_'+flag]).astype(int)
    infile['hot95_mn.21_'+flag+'.d']=(infile[flag]>=infile['hot95_mn.21_'+flag]).astype(int)
    infile['hot99_mn.21_'+flag+'.d']=(infile[flag]>=infile['hot99_mn.21_'+flag]).astype(int)

    infile['cold01_sea.95_'+flag+'.d']=(infile[flag]<=infile['cold01_sea.95_'+flag]).astype(int)
    infile['cold05_sea.95_'+flag+'.d']=(infile[flag]<=infile['cold05_sea.95_'+flag]).astype(int)
    infile['hot95_sea.95_'+flag+'.d']=(infile[flag]>=infile['hot95_sea.95_'+flag]).astype(int)
    infile['hot99_sea.95_'+flag+'.d']=(infile[flag]>=infile['hot99_sea.95_'+flag]).astype(int)
    infile['cold01_sea.1_'+flag+'.d']=(infile[flag]<=infile['cold01_sea.1_'+flag]).astype(int)
    infile['cold05_sea.1_'+flag+'.d']=(infile[flag]<=infile['cold05_sea.1_'+flag]).astype(int)
    infile['hot95_sea.1_'+flag+'.d']=(infile[flag]>=infile['hot95_sea.1_'+flag]).astype(int)
    infile['hot99_sea.1_'+flag+'.d']=(infile[flag]>=infile['hot99_sea.1_'+flag]).astype(int)
    infile['cold01_sea.21_'+flag+'.d']=(infile[flag]<=infile['cold01_sea.21_'+flag]).astype(int)
    infile['cold05_sea.21_'+flag+'.d']=(infile[flag]<=infile['cold05_sea.21_'+flag]).astype(int)
    infile['hot95_sea.21_'+flag+'.d']=(infile[flag]>=infile['hot95_sea.21_'+flag]).astype(int)
    infile['hot99_sea.21_'+flag+'.d']=(infile[flag]>=infile['hot99_sea.21_'+flag]).astype(int)
    
    tempdf=infile[['year']+[i for i in infile.columns if '.d' in i]]
    tempdf=tempdf.groupby('year').sum()
    tempdf['Kreise_Code']=kreise_code
 
    count_df=pd.concat([count_df,tempdf],axis=0)
    
flag_df=pd.DataFrame()
for file in files_list:
    kreise_code=int(file.split('.')[0])
    infile=pd.read_csv(flag_path+file)
    infile=infile[['MESS_DATUM','Kreise_Code',flag,'yyyymm','year','month']]
    infile.to_csv(flag_path+file,index=False)
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    infile['yyyyss']=((infile['year']*10)+infile['season']).astype(int)

    meansea=infile[['yyyyss',flag]].groupby(by='yyyyss').mean().reset_index()
    meansea=meansea.rename(columns={flag:flag+'mean_sea'})
    mediansea=infile[['yyyyss',flag]].groupby(by='yyyyss').median().reset_index()
    mediansea=mediansea.rename(columns={flag:flag+'median_sea'})
    stdsea=infile[['yyyyss',flag]].groupby(by='yyyyss').std().reset_index()
    stdsea=stdsea.rename(columns={flag:flag+'std_sea'})
    
    seadf=pd.merge(meansea,stdsea,on='yyyyss',how='inner')
    seadf['year']=(seadf['yyyyss']/10).astype(int)
    seadf=seadf.drop('yyyyss',axis=1)
    seadf=seadf.groupby('year').mean().reset_index()
    mediansea['year']=(mediansea['yyyyss']/10).astype(int)
    mediansea=mediansea.drop('yyyyss',axis=1)
    mediansea=mediansea.groupby('year').median().reset_index()
    seadf=pd.merge(seadf,mediansea,on='year',how='inner')
    
    meanmn=infile[['yyyymm',flag]].groupby(by='yyyymm').mean().reset_index()
    meanmn=meanmn.rename(columns={flag:flag+'mean_mn'})
    medianmn=infile[['yyyymm',flag]].groupby(by='yyyymm').median().reset_index()
    medianmn=medianmn.rename(columns={flag:flag+'median_mn'})
    stdmn=infile[['yyyymm',flag]].groupby(by='yyyymm').std().reset_index()
    stdmn=stdmn.rename(columns={flag:flag+'std_mn'})
    
    mndf=pd.merge(meanmn,stdmn,on='yyyymm',how='inner')
    mndf['year']=(mndf['yyyymm']/100).astype(int)
    mndf=mndf.drop('yyyymm',axis=1)
    mndf=mndf.groupby('year').mean().reset_index()
    medianmn['year']=(medianmn['yyyymm']/100).astype(int)
    medianmn=medianmn.drop('yyyymm',axis=1)
    medianmn=medianmn.groupby('year').median().reset_index()
    mndf=pd.merge(mndf,medianmn,on='year',how='inner')   

    
    outdf=pd.merge(seadf,mndf,on='year',how='inner')
    outdf['Kreise_Code']=kreise_code
    flag_df=pd.concat([flag_df,outdf],axis=0)

    '''
flag_df=pd.DataFrame()
for file in files_list:
    kreise_code=int(file.split('.')[0])
    infile=pd.read_csv(flag_path+file)
    infile=infile[['MESS_DATUM','Kreise_Code',flag,'yyyymm','year','month']]
    infile.to_csv(flag_path+file,index=False)
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    infile['yyyyss']=((infile['year']*10)+infile['season']).astype(int)
    
    meansea=infile[['yyyyss',flag]].groupby(by='yyyyss').mean().reset_index()
    meansea=meansea.rename(columns={flag:flag+'mean_sea'})
    mediansea=infile[['yyyyss',flag]].groupby(by='yyyyss').median().reset_index()
    mediansea=mediansea.rename(columns={flag:flag+'median_sea'})
    stdsea=infile[['yyyyss',flag]].groupby(by='yyyyss').std().reset_index()
    stdsea=stdsea.rename(columns={flag:flag+'std_sea'})
    
    seadf=pd.merge(meansea,stdsea,on='yyyyss',how='inner')
    seadf['year']=(seadf['yyyyss']/10).astype(int)
    seadf=seadf.drop('yyyyss',axis=1)
    seadf=seadf.groupby('year').mean().reset_index()
    mediansea['year']=(mediansea['yyyyss']/10).astype(int)
    mediansea=mediansea.drop('yyyyss',axis=1)
    mediansea=mediansea.groupby('year').median().reset_index()
    seadf=pd.merge(seadf,mediansea,on='year',how='inner')
    
    meanmn=infile[['yyyymm',flag]].groupby(by='yyyymm').mean().reset_index()
    meanmn=meanmn.rename(columns={flag:flag+'mean_mn'})
    medianmn=infile[['yyyymm',flag]].groupby(by='yyyymm').median().reset_index()
    medianmn=medianmn.rename(columns={flag:flag+'median_mn'})
    stdmn=infile[['yyyymm',flag]].groupby(by='yyyymm').std().reset_index()
    stdmn=stdmn.rename(columns={flag:flag+'std_mn'})
    
    mndf=pd.merge(meanmn,stdmn,on='yyyymm',how='inner')
    mndf['year']=(mndf['yyyymm']/100).astype(int)
    mndf=mndf.drop('yyyymm',axis=1)
    mndf=mndf.groupby('year').mean().reset_index()
    medianmn['year']=(medianmn['yyyymm']/100).astype(int)
    medianmn=medianmn.drop('yyyymm',axis=1)
    medianmn=medianmn.groupby('year').median().reset_index()
    mndf=pd.merge(mndf,medianmn,on='year',how='inner')   
    
    outdf=pd.merge(seadf,mndf,on='year',how='inner')
    outdf['Kreise_Code']=kreise_code
    flag_df=pd.concat([flag_df,outdf],axis=0)
    '''

def seasondf(df,season):
    sumdf=df[['yyyyss',flag]].groupby(by='yyyyss').sum().reset_index()
    sumdf['year']=(sumdf['yyyyss']/10).astype(int)
    sumdf=sumdf.rename(columns={flag:season+flag+'_sum'})
    sumdf=sumdf.drop('yyyyss',axis=1)
    stddf=df[['yyyyss',flag]].groupby(by='yyyyss').std().reset_index()
    stddf['year']=(stddf['yyyyss']/10).astype(int)
    stddf=stddf.rename(columns={flag:season+flag+'_std'})
    stddf=stddf.drop('yyyyss',axis=1)
    maxdf=df[['yyyyss',flag]].groupby(by='yyyyss').max().reset_index()
    maxdf['year']=(maxdf['yyyyss']/10).astype(int)
    maxdf=maxdf.rename(columns={flag:season+flag+'_max'})
    maxdf=maxdf.drop('yyyyss',axis=1)
    mindf=df[['yyyyss',flag]].groupby(by='yyyyss').min().reset_index()
    mindf['year']=(mindf['yyyyss']/10).astype(int)
    mindf=mindf.rename(columns={flag:season+flag+'_min'})
    mindf=mindf.drop('yyyyss',axis=1)
    finddf=pd.merge(sumdf,stddf,on='year',how='inner')
    #finddf=pd.merge(finddf,maxdf,on='year',how='inner') ##comment for precipitation
    #finddf=pd.merge(finddf,mindf,on='year',how='inner')
    return finddf


season_df=pd.DataFrame()
for file in files_list:
    infile=pd.read_csv(flag_path+file)
    outdf=pd.DataFrame()
    kreise_code=int(file.split('.')[0])
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    infile['yyyyss']=((infile['year']*10)+infile['season']).astype(int)
    
    springdf=infile[infile['season']==2]
    summerdf=infile[infile['season']==3]
    autumndf=infile[infile['season']==4]
    winterdf=infile[infile['season']==1]
    
    springdf=seasondf(springdf,season='spring')
    summerdf=seasondf(summerdf,season='summer')
    autumndf=seasondf(autumndf,season='autumn')
    winterdf=seasondf(winterdf,season='winter')
    
    outdf=pd.merge(springdf,summerdf,on='year',how='inner')
    outdf=pd.merge(outdf,autumndf,on='year',how='inner')
    outdf=pd.merge(outdf,winterdf,on='year',how='inner')
    
    outdf['Kreise_Code']=kreise_code
    season_df=pd.concat([season_df,outdf],axis=0)

flag='tas'
flag_path='D://weather_grid_data//kreise_grid_daily_tas//'
files_list=os.listdir(flag_path)    
stress_df=pd.DataFrame()
for file in files_list:
    infile=pd.read_csv(flag_path+file)
    kreise_code=int(file.split('.')[0])
    infile['high25days']=(infile[flag]>=25).astype(int)
    infile['high30days']=(infile[flag]>=30).astype(int)
    #infile['high30.tas']=infile['high30.d']*infile[flag]
    f1=infile[['year','high30days','high25days']].groupby('year').sum().reset_index()
    #f2=infile[infile['high30.tas']>=30]
    #f2=f2[['year','high30.tas']].groupby('year').mean().reset_index()
    #infile=pd.merge(f1,f2,on=['year'],how='inner')
    #infile['Kreise_Code']=kreise_code
    f1['Kreise_Code']=kreise_code
    stress_df=pd.concat([stress_df,f1],axis=0)

econdf=novv5[['year', 'Kreise_Code','lngdp_percap.d1', 'primary_gvapwkr.lg.d1', 'manufacture_gvapwkr.lg.d1',
       'service_gvapwkr.lg.d1']]
econdf1=pd.DataFrame()
kreise_list=list(set(econdf['Kreise_Code']))
for kreise in kreise_list:
    temp=econdf[econdf['Kreise_Code']==kreise].drop('Kreise_Code',axis=1)
    temp=temp.set_index('year')
    temp=temp.shift(-1).reset_index()
    temp['Kreise_Code']=kreise
    econdf1=pd.concat([econdf1,temp],axis=0)



mean_cols=['tasmean_sea']
sum_cols=['prec']
part1=novv7[novv7['year']>=1996]
part1=part1[part1['year']<=1998]
part1=part1[mean_cols+sum_cols+['Kreise_Code']]
part2=novv7[novv7['year']>=2017]
part2=part2[part2['year']<=2019]
part1_means=part1[['Kreise_Code']+mean_cols].groupby(['Kreise_Code']).mean()
part1_sums=part1[['Kreise_Code']+sum_cols].groupby(['Kreise_Code']).sum()
part1=pd.merge(part1_means,part1_sums,on='Kreise_Code',how='inner')
part1['phase']=1
part2_means=part2[['Kreise_Code']+mean_cols].groupby(['Kreise_Code']).mean()
part2_sums=part2[['Kreise_Code']+sum_cols].groupby(['Kreise_Code']).sum()
part2=pd.merge(part2_means,part2_sums,on='Kreise_Code',how='inner')
part2['phase']=2
flag,pal='prec','pink' 
titlesup='Mean Total Precipitation (mm) in German Counties'
part1_map=pd.concat([kreise_file,part1],axis=1)
part1_map=part1_map[['Kreise_Code','geometry',flag]]
part2_map=pd.concat([kreise_file,part2],axis=1)
part2_map=part2_map[['Kreise_Code','geometry',flag]]

fig_map,ax_map=plt.subplots(1,2,figsize=(15,10))
max_flag=part2_map[flag].max()
min_flag=part1_map[flag].min()
part1_map.plot(column=part1_map[flag],ax=ax_map[0],cmap=pal,legend=True,vmax=max_flag,vmin=min_flag)
ax_map[0].set_title('1996-1998',fontsize=20)
ax_map[0].axis('off')
part2_map.plot(column=part2_map[flag],ax=ax_map[1],cmap=pal,legend=True,vmax=max_flag,vmin=min_flag)
ax_map[1].set_title('2017-2019',fontsize=20)
ax_map[1].axis('off')
fig_map.suptitle(titlesup,fontsize=25)
plt.show()

crossdf=crossdf[['Kreise_Code','tasmean_sea.d1','prec.d1']].dropna()
fig,ax=plt.subplots(1,2,figsize=(20,7))
crossdf[crossdf['tasmean_sea.d1']>0]['tasmean_sea.d1'].hist(ax=ax[0],color='red')
crossdf[crossdf['tasmean_sea.d1']<=0]['tasmean_sea.d1'].hist(ax=ax[0],color='green')
ax[0].set_title('Annual Mean Temperature Change (°C)',fontsize=22)
crossdf[crossdf['prec.d1']>0]['prec.d1'].hist(ax=ax[1],color='red')
crossdf[crossdf['prec.d1']<=0]['prec.d1'].hist(ax=ax[1],color='green')
ax[1].set_title('Annual Total Precipitation Change(mm)',fontsize=22)
fig.suptitle('The County-Level Change in Climate between 1996-98 and 2017-19',fontsize=27)
ax[0].tick_params(axis='both', which='major', labelsize=15)
ax[1].tick_params(axis='both', which='major', labelsize=15)
plt.show()

scatter=crossdf[['Kreise_Code','tasmean_sea.d1','prec.d1','lngdp_percap.d1','lnprimary_gvapwkr.d1','lnmanufacture_gvapwkr.d1','lnservice_gvapwkr.d1']]
scatter=scatter.dropna()

unify=pd.read_csv('D://weather_grid_data//model_df//crossdfreg.csv')
unify=unify[['Kreise_Code','West-Germany']].drop_duplicates(keep='first')
scatter=pd.merge(scatter,unify,on='Kreise_Code',how='inner')
'''
scatter=pd.read_csv('D://weather_grid_data//model_df//crossdfreg96.csv')
wg=scatter[scatter['West-Germany']==1]
eg=scatter[scatter['West-Germany']==0]
fig,ax=plt.subplots(1,1,figsize=(15,8.5))
plt.scatter(x=wg['tasmean_sea.d1'],y=wg['lngdp_percap.d1'],color='purple',marker='*')
plt.scatter(x=eg['tasmean_sea.d1'],y=eg['lngdp_percap.d1'],color='purple',marker='x')
plt.scatter(x=wg['tasmean_sea.d1'],y=wg['lnservice_gvapwkr.d1'],color='crimson',marker='*')
plt.scatter(x=eg['tasmean_sea.d1'],y=eg['lnservice_gvapwkr.d1'],color='crimson',marker='x')
plt.scatter(x=wg['tasmean_sea.d1'],y=wg['lnmanufacture_gvapwkr.d1'],color='navy',marker='*')
plt.scatter(x=eg['tasmean_sea.d1'],y=eg['lnmanufacture_gvapwkr.d1'],color='navy',marker='x')
plt.scatter(x=wg['tasmean_sea.d1'],y=wg['lnprimary_gvapwkr.d1'],color='forestgreen',marker='*')
plt.scatter(x=eg['tasmean_sea.d1'],y=eg['lnprimary_gvapwkr.d1'],color='forestgreen',marker='x')
plt.scatter(x=wg['tasmean_sea.d1'],y=wg['lnPopulation.d1'],color='red',marker='*')
plt.scatter(x=eg['tasmean_sea.d1'],y=eg['lnPopulation.d1'],color='red',marker='x')
points = [Line2D([0], [0], label='LnGDP_percap.D1',marker='o', markersize=10,markerfacecolor='purple', linestyle=''),
          Line2D([0], [0], label='LnServiceGVA_prwkr.D1',marker='o', markersize=10,markerfacecolor='crimson', linestyle=''),
          Line2D([0], [0], label='LnManufactureGVA_prwkr.D1', marker='o',markersize=10,markerfacecolor='navy', linestyle=''),
          Line2D([0], [0], label='LnPrimaryGVA_prwkr.D1',marker='o', markersize=10,markerfacecolor='forestgreen', linestyle=''),
          Line2D([0], [0], label='LnPopulation.D1',marker='o', markersize=10,markerfacecolor='red', linestyle=''),
          Line2D([0], [0], label='East-Germany', markersize=10,marker='x',markerfacecolor='darkslategray', linestyle=''),
          Line2D([0], [0], label='West-Germany',marker='*', markersize=10,markerfacecolor='darkslategray', linestyle=''),
          ]
plt.xlabel('Change in Mean Annual Temperature (°C)',fontsize=15)
plt.ylabel('Change in Per-Capita/Worker GDP/GVA (Ln)',fontsize=15)
plt.title('Change in Per-Capita(Worker GDP/GVA vs Change in Medium Term Temperature',fontsize=18)
plt.legend(handles=points,fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.show()
'''
##### SCATTER WITH PANEL 2010 DATA #####
scatter=pd.read_csv('D://weather_grid_data//model_df//modelnovv7.csv'')
wg=scatter[scatter['West-Germany']==1]
eg=scatter[scatter['West-Germany']==0]
fig,ax=plt.subplots(1,3,figsize=(20,12))
ax[0,0].scatter(x=wg['tasmean_sea.d1'],y=wg['service_gvapwkr.lg.d1'],color='navy',marker='*',label='West-Germany')
ax[0,0].scatter(x=eg['tasmean_sea.d1'],y=eg['service_gvapwkr.lg.d1'],color='crimson',marker='x',label='East-Germany')
ax[0,0].set_xlabel('Δ Annual Temperature (°C)',fontsize=15)
ax[0,0].set_ylabel('Δ Service Sector GVA perworker',fontsize=15)
ax[0,0].legend(fontsize=13)
ax[0,0].tick_params(axis='both', which='major', labelsize=12)
ax[0,0].tick_params(axis='both', which='major', labelsize=12)
ax[0,1].scatter(x=wg['tasmean_sea.d1'],y=wg['manufacture_gvapwkr.lg.d1'],color='navy',marker='*',label='West-Germany')
ax[0,1].scatter(x=eg['tasmean_sea.d1'],y=eg['manufacture_gvapwkr.lg.d1'],color='crimson',marker='x',label='East-Germany')
ax[0,1].set_xlabel('Δ Annual Temperature (°C)',fontsize=15)
ax[0,1].set_ylabel('Δ Manufacturing Sector GVA perworker',fontsize=15)
ax[0,1].legend(fontsize=13)
ax[0,1].tick_params(axis='both', which='major', labelsize=12)
ax[0,1].tick_params(axis='both', which='major', labelsize=12)
ax[1,0].scatter(x=wg['tasmean_sea.d1'],y=wg['primary_gvapwkr.lg.d1'],color='navy',marker='*',label='West-Germany')
ax[1,0].scatter(x=eg['tasmean_sea.d1'],y=eg['primary_gvapwkr.lg.d1'],color='crimson',marker='x',label='East-Germany')
ax[1,0].set_xlabel('Δ Annual Temperature (°C)',fontsize=15)
ax[1,0].set_ylabel('Δ Primary Sector GVA perworker',fontsize=15)
ax[1,0].legend(fontsize=13)
ax[1,0].tick_params(axis='both', which='major', labelsize=12)
ax[1,0].tick_params(axis='both', which='major', labelsize=12)
ax[1,1].scatter(x=wg['tasmean_sea.d1'],y=wg['Population.lg.d1'],color='navy',marker='*',label='West-Germany')
ax[1,1].scatter(x=eg['tasmean_sea.d1'],y=eg['Population.lg.d1'],color='crimson',marker='x',label='East-Germany')
ax[1,1].set_xlabel('Δ Annual Temperature (°C)',fontsize=15)
ax[1,1].set_ylabel('Δ Population',fontsize=15)
ax[1,1].legend(fontsize=13)
ax[1,1].tick_params(axis='both', which='major', labelsize=12)
ax[1,1].tick_params(axis='both', which='major', labelsize=12)
fig.suptitle('Change in Economic Activity vs Δ Mean Temperature from 2010 to 2019',fontsize=25)
#plt.legend(handles=points,fontsize=12)
plt.show()

fig,ax=plt.subplots(1,2,figsize=(20,6))
ax[0].scatter(x=wg['tasmean_sea'],y=wg['lngdp_percap.d1'],color='navy',marker='*',label='West-Germany')
ax[0].scatter(x=eg['tasmean_sea'],y=eg['lngdp_percap.d1'],color='crimson',marker='x',label='East-Germany')
ax[0].set_xlabel('Annual Temperature (°C)',fontsize=15)
ax[0].set_ylabel('Δ Ln Per-Capita GDP',fontsize=15)
ax[0].legend(fontsize=13)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].scatter(x=wg['tasmean_sea.d1'],y=wg['lngdp_percap.d1'],color='navy',marker='*',label='West-Germany')
ax[1].scatter(x=eg['tasmean_sea.d1'],y=eg['lngdp_percap.d1'],color='crimson',marker='x',label='East-Germany')
ax[1].set_xlabel('Δ Annual Temperature (°C)',fontsize=15)
ax[1].set_ylabel('Δ Ln Per-Capita GDP',fontsize=15)
ax[1].legend(fontsize=13)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
fig.suptitle('Change in Per-Capita GDP (ln) vs  Temperature (°C) from 2010 to 2019',fontsize=25)
#plt.legend(handles=points,fontsize=12)
plt.show()

## xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx ##

##### SCATTER WITH CROSS-SECTION FROM 1996-98 and 2017-19 #####
def makeline(df,val,deg=1):
    fittas=polyfit(df['tasmean_sea.d1'],df[val],deg=1)
    df[val+'.fit']=(fittas[0]*df['Δ Temperature'])+fittas[1]
    return df
    
scatter=pd.read_csv('D://weather_grid_data/model_df/crossdf2000reg.csv')
vals=['gdp_percap.r.lg.d1','service_gvapwkr.r.lg.d1','manufacture_gvapwkr.r.lg.d1','primary_gvapwkr.r.lg.d1','Population.r.lg.d1']
scatter=scatter[vals+['tasmean_sea.r.lg.d1','West-Germany']]
wg=scatter[scatter['West-Germany']==1]
wg['Δ Temperature']=np.linspace(1.0362,2.7505,wg.shape[0])
eg=scatter[scatter['West-Germany']==0]
eg['Δ Temperature']=np.linspace(1.0362,2.7505,eg.shape[0])
for val in vals:
    wg=makeline(wg,val)
    eg=makeline(eg,val)

fig,ax=plt.subplots(1,3,figsize=(75,22))
ax[0].scatter(x=wg['tasmean_sea.r.lg.d1'],y=wg['gdp_percap.r.lg.d1'],color='navy',label='West-Germany')
ax[0].scatter(x=eg['tasmean_sea.r.lg.d1'],y=eg['gdp_percap.r.lg.d1'],color='crimson',label='East-Germany')
ax[0].plot(wg['Δ Temperature'],wg['gdp_percap.r.lg.d1.fit'],color='navy',linewidth=5)
ax[0].plot(eg['Δ Temperature'],eg['gdp_percap.r.lg.d1.fit'],color='crimson',linewidth=5)
ax[0].set_xlabel('Δ Annual Temperature (°C)',fontsize=axfont)
ax[0].set_ylabel('Δ .GDP per-capita',fontsize=axfont)
ax[0].legend(fontsize=35)
ax[0].tick_params(axis='both', which='major', labelsize=tkfont)
ax[0].tick_params(axis='both', which='major', labelsize=tkfont)
ax[0].set_title('GDP Per-Capita',fontsize=subfont)
ax[1].scatter(x=wg['tasmean_sea.r.lg.d1'],y=wg['service_gvapwkr.r.lg.d1'],color='navy',label='West-Germany')
ax[1].scatter(x=eg['tasmean_sea.r.lg.d1'],y=eg['service_gvapwkr.r.lg.d1'],color='crimson',label='East-Germany')
ax[1].plot(wg['Δ Temperature'],wg['service_gvapwkr.r.lg.d1.fit'],color='navy',linewidth=5)
ax[1].plot(eg['Δ Temperature'],eg['service_gvapwkr.r.lg.d1.fit'],color='crimson',linewidth=5)
ax[1].set_xlabel('Δ Annual Temperature (°C)',fontsize=axfont)
ax[1].set_ylabel('Δ .GVA per-worker',fontsize=axfont)
ax[1].legend(fontsize=35)
ax[1].tick_params(axis='both', which='major', labelsize=tkfont)
ax[1].tick_params(axis='both', which='major', labelsize=tkfont)
ax[1].set_title('Service Sector',fontsize=subfont)
ax[2].scatter(x=wg['tasmean_sea.r.lg.d1'],y=wg['manufacture_gvapwkr.r.lg.d1'],color='navy',marker='*',label='West-Germany')
ax[2].plot(wg['Δ Temperature'],wg['manufacture_gvapwkr.r.lg.d1.fit'],color='navy',linewidth=5)
ax[2].scatter(x=eg['tasmean_sea.r.lg.d1'],y=eg['manufacture_gvapwkr.r.lg.d1'],color='crimson',marker='x',label='East-Germany')
ax[2].plot(eg['Δ Temperature'],eg['manufacture_gvapwkr.r.lg.d1.fit'],color='crimson',linewidth=5)
ax[2].set_xlabel('Δ Annual Temperature (°C)',fontsize=axfont)
ax[2].set_ylabel('Δ .GVA per-worker',fontsize=axfont)
ax[2].legend(fontsize=35)
ax[2].tick_params(axis='both', which='major', labelsize=tkfont)
ax[2].tick_params(axis='both', which='major', labelsize=tkfont)
ax[2].set_title('Manufacturing Sector',fontsize=subfont)
fig.suptitle('Change in Economic Activity vs Δ Mean Temperature between 2000-03 and 2017-19',fontsize=1.15*supfont)
plt.show()
fig,ax=plt.subplots(1,3,figsize=(75,20))
ax[0].scatter(x=wg['tasmean_sea.r.lg.d1'],y=wg['primary_gvapwkr.r.lg.d1'],color='navy',label='West-Germany')
ax[0].scatter(x=eg['tasmean_sea.r.lg.d1'],y=eg['primary_gvapwkr.r.lg.d1'],color='crimson',label='East-Germany')
ax[0].plot(wg['Δ Temperature'],wg['primary_gvapwkr.r.lg.d1.fit'],color='navy',linewidth=5)
ax[0].plot(eg['Δ Temperature'],eg['primary_gvapwkr.r.lg.d1.fit'],color='crimson',linewidth=5)
ax[0].set_xlabel('Δ Annual Temperature (°C)',fontsize=axfont)
ax[0].set_ylabel('Δ .GVA per-worker',fontsize=axfont)
ax[0].legend(fontsize=35)
ax[0].tick_params(axis='both', which='major', labelsize=tkfont)
ax[0].tick_params(axis='both', which='major', labelsize=tkfont)
ax[0].set_title('Primary Sector',fontsize=subfont)
ax[1].scatter(x=wg['tasmean_sea.r.lg.d1'],y=wg['Population.d1'],color='navy',marker='*',label='West-Germany')
ax[1].plot(wg['Δ Temperature'],wg['Population.d1.fit'],color='navy',linewidth=5)
ax[1].scatter(x=eg['tasmean_sea.d1'],y=eg['lnPopulation.d1'],color='crimson',marker='x',label='East-Germany')
ax[1].plot(eg['Δ Temperature'],eg['lnPopulation.d1.fit'],color='crimson',linewidth=5)
ax[1].set_xlabel('Δ Annual Temperature (°C)',fontsize=axfont)
ax[1].set_ylabel('Δ Ln.Population',fontsize=axfont)
ax[1].legend(fontsize=35)
ax[1].tick_params(axis='both', which='major', labelsize=tkfont)
ax[1].tick_params(axis='both', which='major', labelsize=tkfont)
ax[1].set_title('Population',fontsize=subfont)
plt.show()

#plt.legend(handles=points,fontsize=12)
plt.show()

fig,ax=plt.subplots(1,2,figsize=(50,40/3))
ax[0].scatter(x=wg['tasmean_sea'],y=wg['lngdp_percap.d1'],color='navy',marker='*',label='West-Germany')
ax[0].scatter(x=eg['tasmean_sea'],y=eg['lngdp_percap.d1'],color='crimson',marker='x',label='East-Germany')
ax[0].set_xlabel('Annual Temperature (°C)',fontsize=15)
ax[0].set_ylabel('Δ Ln Percapita GDP',fontsize=15)
ax[0].legend(fontsize=13)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[0].tick_params(axis='both', which='major', labelsize=12)
ax[1].scatter(x=wg['tasmean_sea.d1'],y=wg['lngdp_percap.d1'],color='navy',marker='*',label='West-Germany')
ax[1].scatter(x=eg['tasmean_sea.d1'],y=eg['lngdp_percap.d1'],color='crimson',marker='x',label='East-Germany')
ax[1].set_xlabel('Δ Annual Temperature (°C)',fontsize=15)
ax[1].set_ylabel('Δ Ln Percapita GDP',fontsize=15)
ax[1].legend(fontsize=13)
ax[1].tick_params(axis='both', which='major', labelsize=12)
ax[1].tick_params(axis='both', which='major', labelsize=12)
fig.suptitle('Change in Per-Capita GDP (ln) vs  Temperature (°C) between 1996-98 and 2017-19',fontsize=25)
plt.show()

### xxxxxxxxx ######




xvar,yvar='tasmean_sea','tasmean_sea.d1'
fig,ax=plt.subplots(1,1,figsize=(15,8.5))
plt.scatter(x=wg[xvar],y=wg[yvar],color='mediumblue',marker='*',label='West-Germany')
plt.scatter(x=eg[xvar],y=eg[yvar],color='firebrick',marker='x',label='East-Germany')
plt.xlabel('Initial Temperature (°C)',fontsize=15)
plt.ylabel('Change in Temperature (°C)',fontsize=15)
plt.title('Initial Temperature (°C) vs Change in Temperature (°C)',fontsize=18)
plt.legend()
plt.show()


kreise_file=gpd.read_file('D:\\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_file=kreise_file[['Kreise_Code','geometry']]
annualtas=novv7[['year','tasmean_sea']].groupby('year').mean()
annualprec=novv7[['year','prec']].groupby('year').mean()
annualde=pd.concat([annualtas,annualprec],axis=1)
annualde=annualde.reset_index()
annualde=annualde[annualde['year']>=1991]
years=np.arange(1991,2021,step=1)
fittas=np.polyfit(annualde['year'],annualde['tasmean_sea'],deg=3)
annualde['tas-fitted']=(fittas[0]*annualde['year']**3)+(fittas[1]*annualde['year']**2)+(fittas[2]*annualde['year'])+fittas[3]
fitprec=np.polyfit(annualde['year'],annualde['prec'],deg=3)
annualde['prec-fitted']=(fitprec[0]*annualde['year']**3)+(fitprec[1]*annualde['year']**2)+(fitprec[2]*annualde['year'])+fitprec[3]
fig,ax=plt.subplots(1,2,figsize=(55,15))
ax[0].bar(x=annualde['year'],height=annualde['tasmean_sea'],color='crimson')
ax[0].plot(annualde.set_index('year')['tas-fitted'],color='black',linewidth=5)
ax[0].set_title('Annual Mean Temperature (°C) in Germany from 1991 to 2020',fontsize=45)
ax[0].set_ylim(6.5,annualde['tasmean_sea'].max()+0.5)
ax[1].bar(x=annualde['year'],height=annualde['prec'],color='aqua')
ax[1].plot(annualde.set_index('year')['prec-fitted'],color='black',linewidth=5)
ax[1].set_title('Annual Precipitation (mm) in Germany from 1991 to 2020',fontsize=45)
ax[0].tick_params(axis='both', which='major', labelsize=35)
ax[1].tick_params(axis='both', which='major', labelsize=35)
plt.show()

plotdf=pd.merge(kreise_file,crossdfreg96[['Kreise_Code','West-Germany']],on='Kreise_Code',how='inner')
plotdf['West-Germany']=plotdf['West-Germany'].replace(1,'West-Germany')
plotdf['West-Germany']=plotdf['West-Germany'].replace(0,'East-Germany')
fig,ax=plt.subplots(1,1,figsize=(15,6))
kreise_file.boundary.plot(ax=ax,color='darkgrey',alpha=0.3)
kreise_file.plot(ax=ax,color='lightgrey',label='Missing Counties')
plotdf.plot(ax=ax,categorical=True,legend=True,column='West-Germany')
#plotdf[plotdf['West-Germany']==0].plot(ax=ax,color='red',label='East-German Counties')
ax.axis('off')
plt.title('Counties with History from 1995 to 2020',fontsize=12)
#ax.legend()
plt.show()

#%% MAKING CHARTS FOR THE MACRO WORKSHO %%#

supfont,subfont,tkfont,axfont=80,75,50,75
novv7=pd.read_csv('D://weather_grid_data//model_df//modelnovv7.csv')
boxplot=novv7[['Kreise_Code','year','tasmean_sea','tasmean_sea.d1']]
cols=['1991-95','1996-00','2001-05','2006-10','2011-15','2016-20']
boxplotT0=pd.DataFrame(columns=cols)
boxplotD=pd.DataFrame(columns=cols)
kreise_list=list(set(temp['Kreise_Code']))
for item in cols:
    y0=int(item.split('-')[0])
    temp=boxplot[boxplot['year']>=y0]
    temp=temp[temp['year']<=y0+4]
    boxplotT0[item]=list(temp['tasmean_sea'])
    boxplotD[item]=list(temp['tasmean_sea.d1'])
    
boxplotD=boxplotD.dropna()

def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Years')

fig,ax=plt.subplots(1,2,figsize=(55,21))
vx1=ax[0].violinplot(boxplotT0)
vx2=ax[1].violinplot(boxplotD)
for part in vx1['bodies']:
    part.set_facecolor('magenta')
    part.set_alpha(1)
for part in vx2['bodies']:
    part.set_facecolor('yellow')
    part.set_alpha(1)
ax[0].tick_params(axis='both', which='major', labelsize=tkfont)
ax[0].set_title('Annual Mean Temperatures (°C)',fontsize=subfont)
ax[1].tick_params(axis='both', which='major', labelsize=tkfont)
ax[1].set_title('Year-on-Year change in Temperature (°C)',fontsize=subfont)
fig.suptitle('County Level Temperature Dynamics from 1991 to 2020',fontsize=supfont)
for a in ax:
    set_axis_style(a, cols)
#plt.legend(fontsize=25)
plt.show()

fig,ax=plt.subplots(1,2,figsize=(55,21))
ax[0].bar(x=annualde['year'],height=annualde['tasmean_sea'],color='crimson')
ax[0].plot(annualde.set_index('year')['tas-fitted'],color='black',linewidth=5)
ax[0].set_title('Annual Mean Temperature (°C)',fontsize=subfont)
ax[0].set_ylim(6.5,annualde['tasmean_sea'].max()+0.5)
ax[1].bar(x=annualde['year'],height=annualde['prec'],color='aqua')
ax[1].plot(annualde.set_index('year')['prec-fitted'],color='black',linewidth=5)
ax[1].set_title('Annual Precipitation (mm)',fontsize=subfont)
ax[0].tick_params(axis='both', which='major', labelsize=tkfont)
ax[1].tick_params(axis='both', which='major', labelsize=tkfont)
fig.suptitle('Climate Developments in Germany from 1991 to 2020',fontsize=supfont)
plt.show()

kreise_file=gpd.read_file('D:\\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_file=kreise_file[['Kreise_Code','geometry']]
crossreg=pd.read_csv('D://weather_grid_data/model_df/crossdfreg96.csv')
maps=crossreg[['Kreise_Code','year','tasmean_sea','tasmean_sea.d1','prec','prec.d1']]
maps['tasmean1_sea']=maps['tasmean_sea']+maps['tasmean_sea.d1']
maps['prec1']=maps['prec']+maps['prec.d1']
maps=pd.merge(kreise_file,maps,on)

#redoing temperature plot, dropping precipitation
part1=novv7[novv7['year']>=1996]
part1=part1[part1['year']<=1998]
part1=part1[mean_cols+sum_cols+['Kreise_Code']]
part2=novv7[novv7['year']>=2017]
part2=part2[part2['year']<=2019]
part1=part1[['Kreise_Code','tasmean_sea']].groupby(['Kreise_Code']).mean()
part1['phase']=1
part2=part2[['Kreise_Code','tasmean_sea']].groupby(['Kreise_Code']).mean()
part2['phase']=2
flag,pal='tasmean_sea','spring_r' 
titlesup='Mean Annual Temperature (°C) in German Counties'
part1_map=pd.merge(kreise_file,part1,on='Kreise_Code',how='outer')
part1_map=part1_map[['Kreise_Code','geometry',flag]]
part2_map=pd.merge(kreise_file,part2,on='Kreise_Code',how='outer')
part2_map=part2_map[['Kreise_Code','geometry',flag]]

k=0.8976
fig_map,ax_map=plt.subplots(1,2,figsize=(45,25))
max_flag=part2_map[flag].max()
min_flag=part1_map[flag].min()
part1_map.plot(column=part1_map[flag],ax=ax_map[0],cmap=pal,legend=True,vmax=max_flag,vmin=min_flag)
ax_map[0].set_title('1996-1998',fontsize=k*subfont)
ax_map[0].axis('off')
part2_map.plot(column=part2_map[flag],ax=ax_map[1],cmap=pal,legend=True,vmax=max_flag,vmin=min_flag)
ax_map[1].set_title('2017-2019',fontsize=k*subfont)
ax_map[1].axis('off')
fig_map.suptitle(titlesup,fontsize=k*supfont)
plt.show()

crossdf96=pd.read_csv('D://weather_grid_data//model_df//crossdfreg96.csv')

##making the descriptive statistics for the data

from matplotlib.lines import Line2D
legend_elements_subplot1 = [Line2D([0], [0], color='black', linestyle='-', label='Number of Hot Days'),
                            Line2D([0], [0], color='black', linestyle='--', label='Number of Cold Days')]

fulldf=pd.read_csv('D://weather_grid_data/model_df/fulldf/panelreg.csv')

mainkreise={11000:'Berlin',2000:'Hamburg',9162:'München',
            6412:'Frankfurt am Main',5315:'Köln',8111:'Stuttgart',
            5111:'Düsseldorf',14713:'Leipzig'}
county_colors = {11000: 'blue',2000: 'darkgoldenrod',9162: 'green',
    6412: 'red',5315: 'darkorange',8111: 'brown',5111: 'fuchsia',14713: 'darkolivegreen'}
labels={mainkreise[item]:county_colors[item] for item in mainkreise}
kreise_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=25, label=city)
                  for city, color in labels.items()]
seldf=fulldf[fulldf['Kreise_Code'].isin(mainkreise)]

#seldf=seldf[['Kreise_Code','year']+selcols]
colheads={'tas':'Temperature at 2m (°C)',
          'prec':'Precipitation (mm)',
          'precpd':'Precipitation Intensity (mm)'}
selcols=[['tas'],['prec'],['precpd'],['hotday.d','coldday.d']]
supfont,subfont,tkfont,axfont=30,23,18,75

fig,ax=plt.subplots(2,2,figsize=(35,18))
ix=0
for col in selcols:
    for kreise in mainkreise:
        kreise_name=mainkreise[kreise]
        kreise_color=county_colors[kreise]
        tempdf=seldf[seldf['Kreise_Code']==kreise][['year']+col]
        tempdf['year']=tempdf['year'].astype(str)
        #tempdf=tempdf.set_index('year')
        #tempdf=tempdf.index.astype(int)# keep year index as int
        if len(col)==1:
            ax[int(ix/2)][int(ix%2)].plot(tempdf['year'],tempdf[col[0]],color=kreise_color,linewidth=2.2)
            ax[int(ix/2)][int(ix%2)].tick_params(axis='x', rotation=45,labelsize=tkfont)
            ax[int(ix/2)][int(ix%2)].tick_params(axis='y',labelsize=tkfont)
            ax[int(ix/2)][int(ix%2)].set_title(colheads[col[0]],fontsize=subfont)
        else:
            ax[int(ix/2)][int(ix%2)].plot(tempdf['year'],tempdf[col[0]],color=kreise_color,alpha=0.5,label='Hot Days',linewidth=2.2)
            ax[int(ix/2)][int(ix%2)].plot(tempdf['year'],tempdf[col[1]],color=kreise_color,linestyle='--',label='Cold Days',linewidth=2.2)
            ax[int(ix/2)][int(ix%2)].tick_params(axis='x', rotation=45,labelsize=tkfont)
            ax[int(ix/2)][int(ix%2)].tick_params(axis='y',labelsize=tkfont)
            ax[int(ix/2)][int(ix%2)].set_title('Number of Days with Extreme Temperature',fontsize=subfont)
    #ax[int(ix/2)][int(ix%2)].legend()
    ix=ix+1
fig.legend(handles=kreise_handles, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(labels)//2,fontsize=supfont*0.74)
ax[1][1].legend(handles=legend_elements_subplot1, loc='upper right', edgecolor='black')
fig.suptitle('Climate Observations for Major Counties (by Population) from 2000 to 2020',fontsize=supfont)
plt.show()

season_name={1:'winter',2:'spring',3:'summer',4:'autumn'}
kreise=[int(i.split('.')[0]) for i in files_list]
outdf=pd.DataFrame()
for file in files_list:
    infile=pd.read_csv(flag_path+file)
    kreise=(int(file.split('.')[0]))
    infile['year']=(infile['MESS_DATUM']/10000).astype(int)
    infile['month']=((infile['MESS_DATUM']%10000)/100).astype(int)
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    histdf=infile[infile['year']>=1960]
    histdf=histdf[histdf['year']<=1990]
    tempdf=pd.DataFrame()
    for season in range(4):
        season=season+1
        tas=histdf[histdf['season']==season]['tas']
        low,high=tas.quantile(0.05),tas.quantile(0.95)
        tempdf.loc[kreise,season_name[season]+'05']=low
        tempdf.loc[kreise,season_name[season]+'95']=high
    tempdf=tempdf.reset_index()
    outdf=pd.concat([outdf,tempdf],axis=0)
    
outdf=outdf.rename(columns={'index':'Kreise_Code'})
outdf['year']=0
outdf['hist0']=1960
outdf['hist1']=1990    
threshdf=outdf[outdf['year']==0].set_index('Kreise_Code')
outdf1=pd.DataFrame()
for file in file_list:
    infile=pd.read_csv(flag_path+file)
    kreise=(int(file.split('.')[0]))
    select=threshdf[threshdf['Kreise_Code']==kreise]
    infile['year']=(infile['MESS_DATUM']/10000).astype(int)
    infile['month']=((infile['MESS_DATUM']%10000)/100).astype(int)
    infile['season']=infile['month'].apply(lambda x: season_dict[x])
    infile['season_name']=infile['season'].apply(lambda x: season_name[x])

kreisedets=pd.read_excel('D:\\georef-germany-kreis.xlsx',sheet_name='sheet')
fulldf.hvplot.scatter(x='tas', y='D1.GVAREALPWln',hover_cols=['Kreise_Name'])

fulldf=pd.read_csv('D://weather_grid_data/model_df/fulldf/panelreg.csv')
eastdf=fulldf[fulldf['R_East']==1]
westdf=fulldf[fulldf['R_West']==1]
eastdf.to_csv('D://weather_grid_data/model_df/fulldf/eastdf.csv',index=False)
westdf.to_csv('D://weather_grid_data/model_df/fulldf/westdf.csv',index=False)
fulldf['tas'].describe()

df00=fulldf[fulldf['year']==2000]
df00['tas'].describe()
coldkreise=df00[df00['tas']<=9.48]['Kreise_Code'].to_list()
hotkreise=df00[df00['tas']>=10.60]['Kreise_Code'].to_list()
coldf=fulldf[fulldf['Kreise_Code'].isin(coldkreise)]
hotdf=fulldf[fulldf['Kreise_Code'].isin(hotkreise)]
coldf.to_csv('D://weather_grid_data/model_df/fulldf/coldf.csv',index=False)
hotdf.to_csv('D://weather_grid_data/model_df/fulldf/hotdf.csv',index=False)
fulldf['check']=fulldf['R_East']+fulldf['R_West']

klimadf=fulldf[['tas','hotday.d','coldday.d','prec','precpd']]
klimar=klimadf.corr()
cmap = sns.diverging_palette(240, 10, as_cmap=True)
plt.figure(figsize=(10, 8))
sns.heatmap(klimar, annot=True, cmap='coolwarm',center=0, fmt=".2f")
plt.title('Correlation Heatmap of Climate Statistics',fontsize=18)
plt.show()

plt.figure(figsize=(12, 6))
sns.violinplot(x="year", y="tas", data=fulldf)
plt.title('Annual Mean Temperatures for 400 German Counties',fontsize=25)
plt.xlabel('Year',fontsize=20)
plt.ylabel('Temperature (°C) at 2m',fontsize=20)
plt.xticks(rotation=45,fontsize=16)
plt.show()

daysdf=fulldf[['year','hotday.d','coldday.d']].groupby('year').mean()
plt.figure(figsize=(12, 6))
years = [2000+i for i in range(21)]
bar_width = 0.35
index = range(len(years))
plt.bar(index, daysdf['hotday.d'], bar_width, label='Number of Hot Days',color='crimson')
plt.bar([i + bar_width for i in index], daysdf['coldday.d'], bar_width, label='Number of Cold Days',color='navy')
plt.xlabel('Year',fontsize=12)
plt.ylabel('Number of Days with Extreme Temperatures',fontsize=12)
plt.title('Annual Count of Days with Extreme Temperatures',fontsize=18)
plt.xticks([i + bar_width/2 for i in index], years,fontsize=10,rotation=45)
plt.legend()
plt.show()

kreise_file=gpd.read_file('D:\\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_file=kreise_file[['Kreise_Code','geometry']]

phase1df=pd.read_csv('D://weather_grid_data/model_df/cross_sect/phase1.csv')
phase1df=pd.merge(kreise_file,phase1df,on='Kreise_Code')
phase3df=pd.read_csv('D://weather_grid_data/model_df/cross_sect/phase3.csv')
phase3df=pd.merge(kreise_file,phase3df,on='Kreise_Code')

flag='tas'
pal='spring_r'
fig_map,ax_map=plt.subplots(figsize=(8.5,5))
max_flag=phase3df[flag].max()
min_flag=phase1df[flag].min()
phase3df.plot(column=phase3df[flag],ax=ax_map,cmap=pal,legend=True,vmax=max_flag,vmin=min_flag)
ax_map.axis('off')
plt.title('Mean Tempteratures 2016-2020',fontsize=10)
plt.show()



precpd=fulldf[['year','precpd']].groupby('year').mean()
precpd=precpd.reset_index()
fig, ax1 = plt.subplots(figsize=(12, 6))
sns.violinplot(x="year", y="prec", data=fulldf, ax=ax1, color='paleturquoise',alpha=0.6)
ax1.set_xlabel('Year', fontsize=20)
ax1.set_ylabel('Precipitation (mm)', fontsize=20)
ax1.set_title('Annual Precipitation for 400 German Counties', fontsize=18)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, fontsize=16)
ax2 = ax1.twinx()
ax2.plot(precpd['year'].astype(str),precpd['precpd'], color='r', marker='o', linestyle='-')
ax2.set_ylabel('Precipitation per Day (mm/day)', fontsize=20)
plt.show()


flag='prec'
pal='winter_r'
fig_map,ax_map=plt.subplots(figsize=(8.5,5))
phase3df=fulldf[fulldf['year']==2013]
phase3df=pd.merge(kreise_file,phase3df,on='Kreise_Code')
phase3df.plot(column=phase3df[flag],ax=ax_map,cmap=pal,legend=True)
ax_map.axis('off')
plt.title('Annual Average Precipitation (mm) 2016-2020',fontsize=10)
plt.show()

### predicted values
preddf=pd.read_excel('D://paper_1_components//results_table2.xlsx',sheet_name='allhat')
preddf=preddf[['tas','d1tas','tas2','tasxd1tas','ghat','d1gvarealpwln']]
preddf['ghatline']=-0.0179*preddf['tas']+(0.0012*preddf['tas2'])-(0.0015*preddf['tasxd1tas'])


preddfME=preddf[['tas','ghat']]
preddfME['disc.tas']=preddfME['tas'].round().astype(int)
preddfME=preddfME[['disc.tas','ghat']].groupby('disc.tas').mean().reset_index()
preddfME=preddfME.set_index('disc.tas')
preddfME['d1.ghat']=preddfME['ghat'].diff(1)
preddfME=preddfME.reset_index()

fig,ax=plt.subplots(figsize=(15,10))
ax.scatter(preddf['tas'],preddf['d1gvarealpwln'],color='turquoise')
ax.scatter(preddf['tas'],preddf['ghat'],color='crimson',label='Predicted Values')
ax.tick_params(axis='both', which='major', labelsize=16)
ax.legend(fontsize=16)
ax.set_xlabel('Temperature (°C) at 2m', fontsize=20)
ax.set_ylabel('Growth rate of real GVA per-Worker', fontsize=20)

ax2 = ax.twinx()
ax2.plot(preddfME['disc.tas'], preddfME['d1.ghat'], color='blue')
ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.set_ylabel('Marginal Effect of Temperature', fontsize=20)
plt.title('Growth-Rate and Temperature in German Counties 2000-2020',fontsize=25)
plt.show()

df['ghatline']=-0.0179*preddf['tas']+(0.0012*preddf['tas2'])-(0.0015*preddf['tasxd1tas'])
preddf=preddf.sort_values(by='tas')
preddf=preddf.dropna()
preddf['d1tashat']=preddf['tas'].diff(1)
preddf['d1ghat']=preddf['ghat'].diff(1)
preddf=preddf[preddf['d1tashat']>0]
preddf['dg/dt']=preddf['d1ghat']/preddf['d1tashat']

df2000=fulldf[fulldf['year']==2000]
df2000.columns
df2000['GVAREALPW'].describe()
df2000['GVAREALPW'].describe().loc['75%']
rich=df2000['GVAREALPW'].describe().loc['75%']
poor=df2000['GVAREALPW'].describe().loc['25%']
richlist=df2000[df2000['GVAREALPW']>=rich]['Kreise_Code'].to_list()
poorlist=df2000[df2000['GVAREALPW']<=poor]['Kreise_Code'].to_list()
  
richdf=fulldf[fulldf['Kreise_Code'].isin(richlist)]
poordf=fulldf[fulldf['Kreise_Code'].isin(poorlist)]

richdf.to_csv('D://weather_grid_data//model_df//fulldf//richdf.csv',index=False)
poordf.to_csv('D://weather_grid_data//model_df//fulldf//poordf.csv',index=False)

coldf=pd.read_csv('D://weather_grid_data/model_df/fulldf/coldf.csv')
coldlist=list(set(coldf['Kreise_Code']))
hotdf=pd.read_csv('D://weather_grid_data/model_df/fulldf/hotdf.csv')
hotlist=list(set(hotdf['Kreise_Code']))
fulldf['T_Hot']=(fulldf['Kreise_Code'].isin(hotlist)).astype(int)
fulldf['T_Cold']=(fulldf['Kreise_Code'].isin(coldlist)).astype(int)
kreise_file=gpd.read_file('D:\\official_de_shapefiles.gk3.shape.ebenen\\vg5000_ebenen_1231\\VG5000_KRS.shp',crs=4326)
kreise_file['Kreise_Code']=kreise_file['ARS'].astype(int)
kreise_file=kreise_file[['Kreise_Code','geometry']]
fulldf=pd.read_csv('D://weather_grid_data/model_df/fulldf/panelreg.csv')
fulldf=pd.merge(kreise_file,fulldf,on='Kreise_Code')
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', edgecolor='black', label='Former East Germany', alpha=0.7),
    Patch(facecolor='blue', edgecolor='black', label='Former West Germany', alpha=0.7),
    Patch(facecolor='none', edgecolor='black', label='Cold Region', hatch='o'),
    Patch(facecolor='none', edgecolor='black', label='Hot Region', hatch='///')
]
fig, ax = plt.subplots(figsize=(10, 10))
# Plot East German counties in red
fulldf[fulldf['R_East'] ==1].plot(ax=ax, color='red',alpha=0.7)
# Plot West German counties in blue
fulldf[fulldf['R_West'] == 1].plot(ax=ax, color='blue',alpha=0.7)
# Mark T_Cold regions with 'x'
fulldf[fulldf['T_Cold']==1].plot(ax=ax, color='none',edgecolor='black', hatch='o', markersize=50)
# Mark T_Hot regions with 'o'
fulldf[fulldf['T_Hot']==1].plot(ax=ax, color='none', edgecolor='black',hatch='///', markersize=50)
plt.title('East and West German Counties with Hot and Cold Regions',fontsize=18)
plt.legend(handles=legend_elements)
plt.xticks([])
plt.yticks([])
plt.show()


#%% MAKING DIFFERENCE-OLS REGRESSION DATAFRAME 
fulldf=pd.read_csv('D://weather_grid_data/model_df/fulldf/panelreg.csv')
from numpy import log
cols=['Kreise_Code','GVAREALPW','tas','tas2','hotday.d','coldday.d','prec','precpd','CAPPWln']

part1=fulldf[fulldf['year']<=2004][cols]
part1=part1.groupby('Kreise_Code').mean()
part1['GVAREALPWln']=log(part1['GVAREALPW'])
part1['GVAREALPWln0']=part1['GVAREALPWln']
part2=fulldf[fulldf['year']<=2012]
part2=part2[part2['year']>=2008][cols]
part2=part2.groupby('Kreise_Code').mean()
part2['GVAREALPWln']=log(part2['GVAREALPW'])
part2['GVAREALPWln0']=part2['GVAREALPWln']
part3=fulldf[fulldf['year']>=2016][cols]
part3=part3.groupby('Kreise_Code').mean()
part3['GVAREALPWln']=log(part3['GVAREALPW'])

cols=['GVAREALPWln','tas','tas2','hotday.d','coldday.d','prec','precpd','CAPPWln']
diff21=part2[cols]-part1[cols]
diff21=pd.concat([diff21,part1['GVAREALPWln0']],axis=1)
diff32=part3[cols]-part2[cols]
diff32=pd.concat([diff32,part2['GVAREALPWln0']],axis=1)
diff31=part3[cols]-part1[cols]
diff31=pd.concat([diff31,part1['GVAREALPWln0']],axis=1)

diffcols=['GVAREALPWln','tas','tas2','hotday.d','coldday.d','prec','precpd','CAPPWln']
for col in diffcols:
    diff21=diff21.rename(columns={col:'D.'+col})
    diff32=diff32.rename(columns={col:'D.'+col})
    diff31=diff31.rename(columns={col:'D.'+col})

diff21.to_csv('D://weather_grid_data/model_df/longdiff/diff21vFeb1.csv')
diff32.to_csv('D://weather_grid_data/model_df/longdiff/diff32vFeb1.csv')
diff31.to_csv('D://weather_grid_data/model_df/longdiff/diff31vFeb1.csv')

#%% Making the map for hotday coldday ratio and precipitation intensity

df=pd.read_csv("D://archive/Panel_Modelling/Panel_Modelling/panelreg.csv")
df=df[['Kreise_Code','year','tas']]

def make_map_df(indf):
    indf=indf[['Kreise_Code','hotday.d','coldday.d']].groupby('Kreise_Code').sum().reset_index()
    indf['hot/cold_ratio']=indf['hotday.d']/indf['coldday.d']
    return indf
    
part1=df[(df['year']>=2001) & (df['year']<=2005)]
part1=make_map_df(part1)
part2=df[(df['year']>=2016) & (df['year']<=2020)]
part2=make_map_df(part2)
part1=pd.merge(kreise_file,part1,on='Kreise_Code')
part2=pd.merge(kreise_file,part2,on='Kreise_Code')

combined_min = min(part1['hot/cold_ratio'].min(), part2['hot/cold_ratio'].min())
combined_max = max(part1['hot/cold_ratio'].max(), part2['hot/cold_ratio'].max())
norm = mcolors.Normalize(vmin=combined_min, vmax=combined_max)
cmap = cm.OrRd

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
part1.plot(column='hot/cold_ratio', 
           ax=ax1, 
           legend=False, 
           cmap=cmap, 
           norm=norm)
ax1.set_title('2001 to 2005',fontsize=15)
part2.plot(column='hot/cold_ratio', 
           ax=ax2, 
           legend=False, 
           cmap=cmap, 
           norm=norm)
ax2.set_title('2016 to 2020',fontsize=15)
fig.suptitle('The Ratio of Average Number of Hot-Days to Cold-Days',fontsize=18)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.03, pad=0.05)
cbar.set_label('Ratio of Hot-Days to Cold-Days')
plt.show()

df=pd.read_csv("D://archive/Panel_Modelling/Panel_Modelling/panelreg.csv")
df=df[['Kreise_Code','year','precpd']]
part1=df[(df['year']>=2001) & (df['year']<=2005)]
part2=df[(df['year']>=2016) & (df['year']<=2020)]
part1=part1[['Kreise_Code','precpd']].groupby('Kreise_Code').mean().reset_index()
part2=part2[['Kreise_Code','precpd']].groupby('Kreise_Code').mean().reset_index()
part1=pd.merge(kreise_file,part1,on='Kreise_Code')
part2=pd.merge(kreise_file,part2,on='Kreise_Code')
combined_min = min(part1['precpd'].min(), part2['precpd'].min())
combined_max = max(part1['precpd'].max(), part2['precpd'].max())
norm = mcolors.Normalize(vmin=combined_min, vmax=combined_max)
cmap = cm.YlGnBu

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
part1.plot(column='precpd', 
           ax=ax1, 
           legend=False, 
           cmap=cmap, 
           norm=norm)
ax1.set_title('2001 to 2005',fontsize=15)
part2.plot(column='precpd', 
           ax=ax2, 
           legend=False, 
           cmap=cmap, 
           norm=norm)
ax2.set_title('2016 to 2020',fontsize=15)
fig.suptitle('The Average Daily Precipitation in Days with Precipitation',fontsize=18)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.03, pad=0.05)
cbar.set_label('intensity of precipitation')
plt.show()

df=pd.read_csv("D://archive/Panel_Modelling/Panel_Modelling/panelreg.csv")
df=df[['year','Kreise_Code','GVAREALPW','tas']]
part1=df[(df['year']>=2001)&(df['year']<=2005)]
part1=part1.groupby('Kreise_Code').mean().reset_index()
part2=df[(df['year']>=2016)&(df['year']<=2020)]
part2=part2.groupby('Kreise_Code').mean().reset_index()

part1=pd.merge(kreise_file,part1,on='Kreise_Code')
part2=pd.merge(kreise_file,part2,on='Kreise_Code')
combined_min = min(part1['tas'].min(), part2['tas'].min())
combined_max = max(part1['tas'].max(), part2['tas'].max())
norm = mcolors.Normalize(vmin=combined_min, vmax=combined_max)
cmap = cm.spring_r

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
part1.plot(column='tas', 
           ax=ax1, 
           legend=False, 
           cmap=cmap, 
           norm=norm)
ax1.set_title('2001 to 2005',fontsize=15)
part2.plot(column='tas', 
           ax=ax2, 
           legend=False, 
           cmap=cmap, 
           norm=norm)
ax2.set_title('2016 to 2020',fontsize=15)
fig.suptitle('The Annual Mean Temperature for 400 German Counties',fontsize=18)
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.03, pad=0.05)
cbar.set_label('annual mean temperature (°C)')
plt.show()
