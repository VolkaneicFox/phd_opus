# -*- coding: utf-8 -*-
"""
Created on Wed May 29 06:41:03 2024

@author: sukanya mukherjee

This code prepares the following important data sets that contain the survey 
information and the geological information.

1. full survey at the household level -->

"""
import numpy as np
import geopandas as gpd
#import os
#import pickle
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
import pandas as pd
pd.options.mode.chained_assignment = None 
import pyreadstat as ps
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

shpfile=gpd.read_file("G://SME/paper_2/IndiaDrought/drought_atlas/shapefile/DISTRICT_BOUNDARY.shp")
countryshp=gpd.read_file("G://SME/paper_2/IndiaDrought/drought_atlas/shapefile/India_Country_Boundary.shp")
shpfile=shpfile[['ID','geometry']].rename(columns={'ID':'spei_district_code'})
def givedfdetails(df, meta):
    columndict = meta.column_names_to_labels
    columndets = meta.variable_value_labels
    df = df.rename(columns=columndict)
    return [df, columndets]
def makedata(path):
    df,meta=ps.read_sav(path)
    df,cols=givedfdetails(df,meta)
    return [df,cols]
def finalmerge(d1,d2):
    mergecols=[i for i in d1.columns if i in d2.columns]
    for col in mergecols:
        try:
            d1[col]=d1[col].astype(float)
            d2[col]=d2[col].astype(float)
        except:
            d1[col]=d1[col].astype(str)
            d2[col]=d2[col].astype(str)
    return pd.merge(d1,d2,on=mergecols)

cropdict={101:'paddy',102:'jowar',103:'bajra',104:'maize',107:'barley',
          105:'ragi',108:'small_millets',202:'tur',
          401:'sugarcane',606:'banana',1006:'coconut',1009:'soyabean',
          1101:'cotton',106:'wheat',1488:'other_fodder',
          203:'urad',1001:'ground_nut',204:'moong'}
inputdict={1:'cost_seed',2:'cost_seed',3:'cost_seed',4:'cost_seed',5:'cost_seed',
           6:'cost_chemical_fertilizer',7:'cost_bio_fertlizer',8:'cost_manure',
           9:'cost_chemical_pesticides',10:'cost_bio_pesticides',11:'cost_diesel',
           12:'cost_electricity',13:'cost_irrigation',14:'cost_labour_human',
           15:'cost_labour_animal',16:'cost_minor_machine_repairs',
           17:'cost_interest_on_crop_loan',18:'cost_crop_machinery',
           19:'cost_crop_insurance',20:'cost_land_lease',21:'cost_other',22:'cost_total'}
inputsellerdict= {1: "local_market",2: "apmc_market",3: "input_dealers",
                  4: "cooperative",5: "govt_agencies",6: "fpo",7: "private_processors",
                  8: "contract_farming",10: "own_farm",9: "others"}
inputqualitydict={1:'good',2:'average',3:'poor',4:'dont_know'}
buyersdict={1:'local_market',2:'APMC_market',3:'input_dealers',
            4:'coops',5:'govt_agency',6:'FPO', 7:'pvt_processor',
            8:'contract',9:'otherbuyers'}
irrigationdict={1:'canal',2:'msw',3:'gw',4:'mixed',9:'others',0:'unknown',
                99:'only_one_source'} #msw: minor_surface_works, gw: ground_water
owndict={'owned and possessed':'o&d',
         'leased in':'ls_in','leased_out':'ls_out',
         'otherwise possessed':'x&p','total land':''}
landdict={'land_type':{6:'homestead',7:'homestead',8:'homestead',9:'homestead', 10:'total land',
                       1:'non-homestead',2:'non-homestead',3:'non-homestead',4:'non-homestead',5:'non-homestead'},
          'ownership':{1:'owned and possessed',6:'owned and possessed',5:'leased_out',
                       2:'leased in',3:'leased in',7:'leased in',8:'leased in',
                       4:'otherwise possessed',9:'otherwise possessed',
                       10:'total land'}}

#%% Situation of Agriculture July 2018 to June 2019 -- LAND USE AND OWNERSHIP

landv1df,landv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L4B5land_details_July_Dec_2018.sav")
landv1cols={'Common Primary Key for household identification':'HHID','Srl No.':'land_ownership',
            'Sector':'type','State':'state','District':'nss_district_code19',
            'area of land (0.00 acre)':'land_acre','Visit number':'visit_no',
       'whether used for any agricultural production during July- December 2018':'if_used_agri_v1',
       'land used for shifting/ jhum cultivation':'jhum_acre',
       'land other than the land used for shifting /jhum cultivation':'non_jhum_acre',
       'only for farming of animals':'only_animal_farming',
       'both for growing of crop and farming of animals':'crops_and_animals',
       'other agricultural uses':'other_agri_uses',
       'other land not used for agriculture purpose':'other_non_agri',
       'major type of crop grown/ animal farming undertaken (code)':'farming_output_code',
       'whether any part of the land was rrigated':'if_land_irrigated',
       'area of land irrigated (0.00 acre)':'acre_irrigated',
       'source of irrigation: major source':'primary_irrigation',
       'source of irrigation: 2nd major source':'second_irrigation'}
landv1df=landv1df[list(landv1cols.keys())].rename(columns=landv1cols)
landv1df['nss_district_code19']=(landv1df['state'].astype(int)*100)+landv1df['nss_district_code19'].astype(int)
landv1df['HHID_dt']=(landv1df['HHID'].astype(float)*10000)+landv1df['nss_district_code19']
landv1df['HHID_dt']=landv1df['HHID_dt'].apply(lambda x: '{:.0f}'.format(x))
landv1df['HHID_dt']=landv1df['HHID_dt'].apply(lambda x: str(x)[:-5]+str(x)[-4:])
#landv1df=landv1df[['HHID_dt','land_ownership','if_used_agri_v1','jhum_acre','non_jhum_acre','land_acre',
#                           'if_land_irrigated','acre_irrigated',
#                           'primary_irrigation','second_irrigation',
#                           'farming_output_code','nss_district_code19']]
landv1df['acre_irrigated']=landv1df['acre_irrigated'].fillna(0)
landv1df['primary_irrigation']=landv1df['primary_irrigation'].replace('',np.nan)
landv1df['second_irrigation']=landv1df['second_irrigation'].replace('',np.nan)
landv1df['primary_irrigation']=landv1df['primary_irrigation'].fillna(0)
landv1df['primary_irrigation']=landv1df['primary_irrigation'].astype(float)
landv1df['primary_irrigation']=landv1df['primary_irrigation'].apply(lambda x: irrigationdict[x])
landv1df['second_irrigation']=landv1df['second_irrigation'].fillna(0)
landv1df['second_irrigation']=landv1df['second_irrigation'].astype(float)
landv1df['second_irrigation']=landv1df['second_irrigation'].apply(lambda x: irrigationdict[x])  
landv1df['find_duplicates']=landv1df.groupby('HHID_dt')['HHID_dt'].transform('count')
landv1df['farming_output_code']=landv1df['farming_output_code'].replace('',np.nan).astype(float)
landv1df['land_ownership']=landv1df['land_ownership'].replace('',np.nan).astype(float)
landv1df['land_type']=landv1df['land_ownership'].apply(lambda x:landdict['land_type'][int(x)])
landv1df['land_own']=landv1df['land_ownership'].apply(lambda x: landdict['ownership'][int(x)])
landv1df['acre_total_use']=landv1df[['jhum_acre',
       'non_jhum_acre', 'only_animal_farming', 'crops_and_animals',
       'other_agri_uses', 'other_non_agri']].sum(axis=1)
landv1df['acre_rainfed_total']=np.round(landv1df['acre_total_use']-landv1df['acre_irrigated'],2)
landv1df['acre_crop_use']=landv1df[['jhum_acre', 'non_jhum_acre',
       'crops_and_animals']].sum(axis=1)
landv1df['acre_agri_use1']=landv1df[['jhum_acre',
       'non_jhum_acre', 'only_animal_farming', 'crops_and_animals']].sum(axis=1)
landv1df['acre_agri_use2']=landv1df[['jhum_acre',
       'non_jhum_acre', 'only_animal_farming', 'crops_and_animals','other_agri_uses']].sum(axis=1)
landv1df['acre_rainfed_agri_use1']=landv1df['acre_agri_use1']-landv1df['acre_irrigated']
landv1df['acre_rainfed_crop_use']=landv1df['acre_crop_use']-landv1df['acre_irrigated']
## there are 2067 leased_out and we can remove these observations since the 
#only purpose of this to earn income from being leased out
landv1df=landv1df[landv1df['land_own']!='leased_out']

landv2df,landv2meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V2L4B5land_details_Jan_Jun_2019.sav")
landv2cols={'Common Primary Key for household identification':'HHID','Srl No.':'land_ownership',
            'Sector':'type','State':'state','District':'nss_district_code19',
            'area of land (0.00 acre)':'land_acre','Visit number':'visit_no',
       'whether used for any agricultural production during July- December 2018':'if_used_agri_v1',
       'land used for shifting/ jhum cultivation':'jhum_acre',
       'land other than the land used for shifting /jhum cultivation':'non_jhum_acre',
       'only for farming of animals':'only_animal_farming',
       'both for growing of crop and farming of animals':'crops_and_animals',
       'other agricultural uses':'other_agri_uses',
       'other land not used for agriculture purpose':'other_non_agri',
       'major type of crop grown/ animal farming undertaken (code)':'farming_output_code',
       'whether any part of the land was rrigated':'if_land_irrigated',
       'area of land irrigated (0.00 acre)':'acre_irrigated',
       'source of irrigation: major source':'primary_irrigation',
       'source of irrigation: 2nd major source':'second_irrigation'}
landv2df=landv2df[list(landv2cols.keys())].rename(columns=landv2cols)
landv2df['nss_district_code19']=(landv2df['state'].astype(int)*100)+landv2df['nss_district_code19'].astype(int)
landv2df['HHID_dt']=(landv2df['HHID'].astype(float)*10000)+landv2df['nss_district_code19']
landv2df['HHID_dt']=landv2df['HHID_dt'].apply(lambda x: '{:.0f}'.format(x))
landv2df['HHID_dt']=landv2df['HHID_dt'].apply(lambda x: str(x)[:-5]+str(x)[-4:])
#landv2df=landv2df[['HHID_dt','land_ownership','if_used_agri_v1','jhum_acre','non_jhum_acre','land_acre',
#                           'if_land_irrigated','acre_irrigated',
#                           'primary_irrigation','second_irrigation',
#                           'farming_output_code','nss_district_code19']]
landv2df['acre_irrigated']=landv2df['acre_irrigated'].fillna(0)
landv2df['primary_irrigation']=landv2df['primary_irrigation'].replace('',np.nan)
landv2df['second_irrigation']=landv2df['second_irrigation'].replace('',np.nan)
landv2df['primary_irrigation']=landv2df['primary_irrigation'].fillna(0)
landv2df['primary_irrigation']=landv2df['primary_irrigation'].astype(float)
landv2df['primary_irrigation']=landv2df['primary_irrigation'].apply(lambda x: irrigationdict[x])
landv2df['second_irrigation']=landv2df['second_irrigation'].fillna(0)
landv2df['second_irrigation']=landv2df['second_irrigation'].astype(float)
landv2df['second_irrigation']=landv2df['second_irrigation'].apply(lambda x: irrigationdict[x])  
landv2df['find_duplicates']=landv2df.groupby('HHID_dt')['HHID_dt'].transform('count')
landv2df['farming_output_code']=landv2df['farming_output_code'].replace('',np.nan).astype(float)
landv2df['land_ownership']=landv2df['land_ownership'].replace('',np.nan).astype(float)
landv2df['land_type']=landv2df['land_ownership'].apply(lambda x:landdict['land_type'][int(x)])
landv2df['land_own']=landv2df['land_ownership'].apply(lambda x: landdict['ownership'][int(x)])
landv2df['acre_total_use']=landv2df[['jhum_acre',
       'non_jhum_acre', 'only_animal_farming', 'crops_and_animals',
       'other_agri_uses', 'other_non_agri']].sum(axis=1)
landv2df['acre_rainfed_total']=np.round(landv2df['acre_total_use']-landv2df['acre_irrigated'],2)
landv2df['acre_crop_use']=landv2df[['jhum_acre', 'non_jhum_acre',
       'crops_and_animals']].sum(axis=1)
landv2df['acre_agri_use1']=landv2df[['jhum_acre',
       'non_jhum_acre', 'only_animal_farming', 'crops_and_animals']].sum(axis=1)
landv2df['acre_agri_use2']=landv2df[['jhum_acre',
       'non_jhum_acre', 'only_animal_farming', 'crops_and_animals','other_agri_uses']].sum(axis=1)
landv2df['acre_rainfed_agri_use1']=landv2df['acre_agri_use1']-landv2df['acre_irrigated']
landv2df['acre_rainfed_crop_use']=landv2df['acre_crop_use']-landv2df['acre_irrigated']
## there are 2067 leased_out and we can remove these observations since the 
#only purpose of this to earn income from being leased out
landv2df=landv2df[landv2df['land_own']!='leased_out']

landuse=pd.concat([landv1df,landv2df],axis=0)
landuse.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/landuse_TOTAL_Jan2025.csv",index=False)

## ownership code is shifted to oldcode

#%% Situation of Agriculture July 2018 to June 2019 -- OUTPUT AND SALE
'''
output is measured for households with earnings from self-employment in agriculture
is more than Rs. 4000 for the last 365 days, this is collected in Visit 1 - so the period
of reference is from June 2017 since Visit 1 is canvassed from June 2018 to Dec 2018. 
This is done to only select agricultural households and the actual data on output
and sales is for the reference period July2018 to Dec2018 
'''

outputv1df,outputv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L7B6output_crops_July_Dec_2018.sav")
outputv1df=outputv1df.rename(columns={'Common Primary Key for household identification':'HHID','State':'state',
                                      'State_District':'nss_district_code19','Srl No.':'output_crop_rank',
                                       'Crop code':'crop_code','Unit code':'unit'})
outputv1df['HHID_dt']=(outputv1df['HHID'].astype(float)*10000)+outputv1df['nss_district_code19'].astype(float)
outputv1df['HHID_dt'] = outputv1df['HHID_dt'].apply(lambda x: f'{x:.0f}')
outputv1df['HHID_dt'] = outputv1df['HHID_dt'].astype(str)
output2v1df,output2v1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L6B6output_crops_July_Dec_2018.sav")
output2v1df=output2v1df.rename(columns={'Common Primary Key for household identification':'HHID','State':'state',
                                      'State_District':'nss_district_code19','Srl No.':'output_crop_rank',
                                       'Crop code':'crop_code','Unit code':'unit'})
output2v1df['HHID_dt']=(output2v1df['HHID'].astype(float)*10000)+output2v1df['nss_district_code19'].astype(float)
output2v1df['HHID_dt'] = output2v1df['HHID_dt'].apply(lambda x: f'{x:.0f}')
output2v1df['HHID_dt'] = output2v1df['HHID_dt'].astype(str)
outputTv1df=pd.merge(outputv1df,output2v1df,on=['HHID_dt','crop_code'])
for col in outputTv1df.columns:
    if '_x' in col:
        outputTv1df=outputTv1df.rename(columns={col:col[:-2]})
        coly=col[:-2]+'_y'
        outputTv1df=outputTv1df.drop(coly,axis=1)
    else:
        outputTv1df=outputTv1df.rename(columns={col:col})
## outputTv1df.shape = (109920, 44)
#outputTv1df['HHID_dt'] = outputTv1df['HHID_dt'].apply(lambda x: f'{x:.0f}')
#outputTv1df['HHID_dt'] = outputTv1df['HHID_dt'].astype(str)
##outputTv1df.shape = (109920,44)

outputv1cols={'HHID_dt':'HHID_dt','Sector':'type','nss_district_code19':'nss_district_code19',
'other disposal: sale value (Rs.)':'other_disposal_sale_Rs', 'crop_code':'crop_code', 'unit':'unit',
'output_crop_rank':'output_crop_rank',
'all disposal: quantity sold':'total_output_sale_kg',
'all disposal: sale value (Rs.)':'all_disposal_sale_Rs',  ##major sale + other sale 
'value of pre-harvest sale (Rs.)':'pre_harvest_value_Rs', 
'value of harvested produce (Rs.)':'harvest_value_Rs',
'value of by-products (Rs.)':'value_by_products_Rs', 
'total value (Rs.)':'total_value_Rs',
'area of irrigated land (0.00 acre)':'irrigated_acre','quantity produced from irrigated land':'output_irrigated_acre',
'area of un-irrigated land (0.00 acre)':'unirrigated_acre',
       'quantity produced from un-irrigated land':'output_unirrigated_acre', 
'total quantity':'total_output_quantity','"area of land under pre-harvest sale (0.00 acre)':'preharvest_acre',
       'major disposal-to whom you sold':'major_buyer',
       'are you satisfied with the sale outcome':'sale_satistifaction',
       'major disposal: quantity sold':'major_quantity_sold', 'major disposal: sale value (Rs.)':'major_sale_Rs.',
       'other disposal: quantity sold':'other_quantity_sold'}
outputTv1df=outputTv1df[list(outputv1cols.keys())].rename(columns=outputv1cols)
outputTv1df=outputTv1df[output2v1df['crop_code'].isin(['5999','9999',''])==False]
outputTv1df['crop_code']=outputTv1df['crop_code'].astype(int)
outputTv1df['farming_output_code']=outputTv1df['crop_code']//100
outputTv1df=outputTv1df[outputTv1df['crop_code'].isin(cropdict.keys())]
## outputTv1df.shape = (50071, 25)

#### REPEAT FROM HERE WITH outputTv1df

print('outputTv1df.shape[0] = ',outputTv1df.shape[0])
## outputTv1df.shape = 47881
rows_with_nan = outputTv1df[outputTv1df['irrigated_acre'].isna() & outputTv1df['unirrigated_acre'].isna()]
print('rows_with_nan.shape[0] = ',rows_with_nan.shape[0])

'''
there are 110 rows of observations in which both 'irrigated_acre' and 'unirrigated_acre' are nan,
There are the following issues:
    1. farmer households have different 'total_output_quantity' and 'total_output_sale_kg'; 
       farmers can be selling stock output not cultivatd this period. 
    2. another situation of error is when ('output_irrigated_acre'>0 AND 'irrigated_acre'=nan) OR 
    ('output_unirrigated_acre'>0 AND 'unirrigated_acre'=nan) ---> there are 7+5 rows of observations with 
    this problem.
    THEREFORE, we can drop these households
'''
outputTv1df['stock_output_sale_kg']=outputTv1df['total_output_sale_kg']-outputTv1df['total_output_quantity']
'''
stock_output_sale_kg is the inventory formation, farmers can choose to save their current output for
sale later, or they can sell old stock in the present period. 
# (qty['stock_output_sale_kg']<0).astype(int).sum() = 18770
# (qty['stock_output_sale_kg']==0).astype(int).sum() = 10137
# (qty['stock_output_sale_kg']>0).astype(int).sum() = 412
There are 18770 rows where the farmers are keeping output for later sale and 412 where they are using
previous inventory for sale.
Therefore, present yield is calculated based on 'total_output_quantity'.
'''

outputTv1df['output_nan']= ((outputTv1df['output_irrigated_acre']>0) & outputTv1df['irrigated_acre'].isna()).astype(int) + ((outputTv1df['output_unirrigated_acre']>0) & outputTv1df['unirrigated_acre'].isna()).astype(int)
print(outputTv1df['output_nan'].sum()) #12
outputTv1df=outputTv1df[outputTv1df['output_nan']==0]
print(outputTv1df.shape) # (50059,27)

'''
Next, we consider that 'total_output_sale_kg' indicates the receipts from the sale of output,
and 'total_output_quantity' is the amount cultivated. There are 18550 rows with nan values in 
'total_output_sale_kg' and 438 rows with nan values in 'total_output_quantity'.
# outputTv1df['total_output_sale_kg'].isna().astype(int).sum() = 18550
# outputTv1df['total_output_quantity'].isna().astype(int).sum() = 438
This means, there is the concern that farmer did not sell his output even when he had output.
One thing to potentially consider, is if the sale is reported in the second visit.
'''
# ** visit 2

outputv2df,outputv2meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V2L7B6output_crops_Jan_Jun_2019.sav")
outputv2df=outputv2df.rename(columns={'Common Primary Key for household identification':'HHID','State':'state',
                                      'State_District':'nss_district_code19','Srl No.':'output_crop_rank',
                                       'Crop code':'crop_code','Unit code':'unit'})
outputv2df['HHID_dt']=(outputv2df['HHID'].astype(float)*10000)+outputv2df['nss_district_code19'].astype(float)
outputv2df['HHID_dt'] = outputv2df['HHID_dt'].apply(lambda x: f'{x:.0f}')
outputv2df['HHID_dt'] = outputv2df['HHID_dt'].astype(str) ### outputv2df.shape[0] = 88791
output2v2df,output2v2meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V2L6B6output_crops_Jan_Jun_2019.sav")
output2v2df=output2v2df.rename(columns={'Common Primary Key for household identification':'HHID','State':'state',
                                      'State_District':'nss_district_code19','Srl No.':'output_crop_rank',
                                       'Crop code':'crop_code','Unit code':'unit'})
output2v2df['HHID_dt']=(output2v2df['HHID'].astype(float)*10000)+output2v2df['nss_district_code19'].astype(float)
output2v2df['HHID_dt'] = output2v2df['HHID_dt'].apply(lambda x: f'{x:.0f}')
output2v2df['HHID_dt'] = output2v2df['HHID_dt'].astype(str) ### output2v2df.shape[0] = 88791
outputTv2df=pd.merge(outputv2df,output2v2df,on=['HHID_dt','crop_code'])
for col in outputTv2df.columns:
    if '_x' in col:
        outputTv2df=outputTv2df.rename(columns={col:col[:-2]})
        coly=col[:-2]+'_y'
        outputTv2df=outputTv2df.drop(coly,axis=1)
    else:
        outputTv2df=outputTv2df.rename(columns={col:col})
## outputTv2df.shape = (88791,44)

outputv2cols={'HHID_dt':'HHID_dt','Sector':'type','nss_district_code19':'nss_district_code19',
'other disposal: sale value (Rs.)':'other_disposal_sale_Rs', 'crop_code':'crop_code','unit':'unit',
'output_crop_rank':'output_crop_rank',
'all disposal: quantity sold':'total_output_sale_kg',
'all disposal: sale value (Rs.)':'all_disposal_sale_Rs',  ##major sale + other sale 
'value of pre-harvest sale (Rs.)':'pre_harvest_value_Rs', 
'value of harvested produce (Rs.)':'harvest_value_Rs',
'value of by-products (Rs.)':'value_by_products_Rs', 
'total value (Rs.)':'total_value_Rs',
'area of irrigated land (0.00 acre)':'irrigated_acre','quantity produced from irrigated land':'output_irrigated_acre',
'area of un-irrigated land (0.00 acre)':'unirrigated_acre',
       'quantity produced from un-irrigated land':'output_unirrigated_acre', 
'total quantity':'total_output_quantity','area of land under pre-harvest sale (0.00 acre)':'preharvest_acre',
       'major disposal-to whom you sold':'major_buyer',
       'are you satisfied with the sale outcome':'sale_satistifaction',
       'major disposal: quantity sold':'major_quantity_sold', 'major disposal: sale value (Rs.)':'major_sale_Rs.',
       'other disposal: quantity sold':'other_quantity_sold'}
outputTv2df=outputTv2df[list(outputv2cols.keys())].rename(columns=outputv2cols)
outputTv2df=outputTv2df[output2v2df['crop_code'].isin(['5999','9999',''])==False]
outputTv2df['crop_code']=outputTv2df['crop_code'].astype(int)
outputTv2df['farming_output_code']=outputTv2df['crop_code']//100
outputTv2df=outputTv2df[outputTv2df['crop_code'].isin(cropdict.keys())]
print(outputTv2df.shape)# (31157,25)

rows_with_nan2 = outputTv2df[outputTv2df['irrigated_acre'].isna() & outputTv2df['unirrigated_acre'].isna()]
print('rows_with_nan2.shape[0] = ',rows_with_nan2.shape[0]) ### 128

'''
NOTE: DIFFERENCE BETWEEN HARVESTED VALUE AND SALE VALUE

Block 6 of the survey data contains details of total quantity harvested, the income from sale
and the total value of the harvest. 
Block 6, Level 6, contains the output harvest data, and in output2v1df['total_quantity'] is 
(reported in survey to be) the sum of output_irrigated_acre and output_unirrigated_acre.
Then, Block 6, Level 7, contains all the relevant information regarding sale
'''

outputTv2df['stock_output_sale_kg']=outputTv2df['total_output_sale_kg']-outputTv2df['total_output_quantity']
print((outputTv2df['stock_output_sale_kg']<0).astype(int).sum()) ## = 15209
print((outputTv2df['stock_output_sale_kg']==0).astype(int).sum())## = 4296
print((outputTv2df['stock_output_sale_kg']>0).astype(int).sum()) ## = 199
outputTv2df['output_nan']= ((outputTv2df['output_irrigated_acre']>0) & outputTv2df['irrigated_acre'].isna()).astype(int) + ((outputTv2df['output_unirrigated_acre']>0) & outputTv2df['unirrigated_acre'].isna()).astype(int)
print(outputTv2df['output_nan'].sum()) # 22
outputTv2df=outputTv2df[outputTv2df['output_nan']==0]
print(outputTv2df.shape) #(31135,27)
outputTv1df['visit_no']=outputTv1df['HHID_dt'].apply(lambda x: str(x)[-5])
outputTv2df['visit_no']=outputTv2df['HHID_dt'].apply(lambda x: str(x)[-5])
## we need to remove the visit_no from the HHID_dt ##
outputTv1df['HHID_dt']=outputTv1df['HHID_dt'].apply(lambda x: str(x)[:-5]+str(x)[-4:])
outputTv2df['HHID_dt']=outputTv2df['HHID_dt'].apply(lambda x: str(x)[:-5]+str(x)[-4:])

hhv1=set(outputTv1df['HHID_dt'])
hhv2=set(outputTv2df['HHID_dt'])
print('Number of households in first visit =',len(hhv1)) #38663
print('Number of households in second visit =',len(hhv2)) #25717
print('The number of households common to list of both visits =',len(list(hhv1.intersection(hhv2)))) #25110

outputTv1df['visit_no']=1
outputTv2df['visit_no']=2
outputTdf=pd.concat([outputTv1df,outputTv2df],axis=0)
outputTdf=outputTdf[outputTdf['unit']=='1'] ##unit code=kg ##shape = 78002,28


'''
we need to consider the outliers in the column of 'total_output_sale_kg','all_disposal_sale_Rs'
we also need to consider the point that households might have positive 'total_output_quantity' but have not sold
we have also combined the data for visit1 and visit2

There are two major FACETS to this problem:
    1. The quantity of output produced by the farming household, 'total_output_quantity'
    2. The quantity of output sold by the farming household, 'total_output_sale_kg'
'''
print(outputTv1df[outputTv1df['total_output_sale_kg'].isna()].shape[0]) ## 18550
print(outputTv1df[outputTv1df['all_disposal_sale_Rs'].isna()].shape[0]) ## 18550
print(outputTv2df[outputTv2df['total_output_sale_kg'].isna()].shape[0]) ## 10835
print(outputTv2df[outputTv2df['all_disposal_sale_Rs'].isna()].shape[0]) ## 10835
print(outputTdf[outputTdf['total_output_sale_kg'].isna()].shape[0]) ## 28214
print(outputTdf[outputTdf['all_disposal_sale_Rs'].isna()].shape[0]) ## 28214
'''
since we have removed all situations in which (irrigated_acre=0 AND output_irrigated_acre>0)
and (unirrgated_acre=0 AND output_unirrigated_acre>0), we can fillna(0) for 'irrigated_acre'
and for 'unirrigated_acre'

'total_output_sale_kg' ==> **Quantity pertaining to pre-harvest sale, crop loss/ damaged 
                            (even if available) is to be excluded.**
Therefore, the sample is restricted only to households that have been able to make sale
and it is unclear if the nan/missing values reflect that the household has not made any sale.

Therefore, there are two things to consider:
    1. Sale revenue from the households that reported sale
    2. Value of harvest for all the households that had harvest
    
This data does not have the information on crop-loss or crop-damaged and it cannot be
combined because the survey does not include the information on crop lost with output 
data. Also, we do not know why there is a difference between 'total_output_quantity' 
cultivated by the farmer, and the 'total_output_sale_Rs'. 
For example, if sale was inadequate, or if stock/inventory was used.
'''
outputTv1df['irrigated_acre']=outputTv1df['irrigated_acre'].fillna(0)
outputTv1df['unirrigated_acre']=outputTv1df['unirrigated_acre'].fillna(0)
outputTv2df['irrigated_acre']=outputTv2df['irrigated_acre'].fillna(0)
outputTv2df['unirrigated_acre']=outputTv2df['unirrigated_acre'].fillna(0)
outputTdf['irrigated_acre']=outputTdf['irrigated_acre'].fillna(0)
outputTdf['unirrigated_acre']=outputTdf['unirrigated_acre'].fillna(0)
outputTdf['irrigated_m2']=outputTdf['irrigated_acre']*4046.86
outputTdf['unirrigated_m2']=outputTdf['unirrigated_acre']*4046.86

outputTdf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/outputhouseholds-vJan25.csv",index=False)

#%% ------- ****** ------ Combining output with irrigation data ------ ****** ---------

outputTdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/outputhouseholds-vJan25.csv")
landdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/landuse_TOTAL_Jan2025.csv")

# we want to homestead and non-homestead land since some homestead land is used in cultivation

landdf=landdf[landdf['land_type']!='total land'] ## prevent double counting
landdf=landdf[['HHID_dt','visit_no','farming_output_code','land_own','land_acre',
               'if_land_irrigated','primary_irrigation','second_irrigation']].drop_duplicates().dropna(subset='farming_output_code') ## only farmed land is selected

# for each hhld and each visit and each farming_output_code, find the share of land that
# is irrigated and also share that has different types of ownership

print('The number of households in outputTdf = ',outputTdf.HHID_dt.nunique())
print('The number of households in landdfT = ',landdf.HHID_dt.nunique())
print('The number of households in both = ',len(set(landdf.HHID_dt).intersection(set(outputTdf.HHID_dt))))

'''
The number of households in outputTdf =  39625
The number of households in landdfT =  47541
The number of households in both =  39550
'''
hhlds=list(set(landdf.HHID_dt).intersection(set(outputTdf.HHID_dt)))
outputTdf=outputTdf[outputTdf['HHID_dt'].isin(hhlds)]
landdf=landdf[landdf['HHID_dt'].isin(hhlds)]
landdf=landdf[landdf['farming_output_code'].isin(outputTdf['farming_output_code'].unique())]
landdf=landdf.drop_duplicates()

landown=landdf[['HHID_dt','farming_output_code','visit_no','land_own','land_acre']].drop_duplicates()
landown['total_land']=landown[['HHID_dt','farming_output_code','visit_no','land_acre']].groupby(['HHID_dt','farming_output_code','visit_no']).transform('sum')
landown['land_own_share']=landown['land_acre']/landown['total_land']
landown=landown[['HHID_dt','farming_output_code','visit_no','land_own','land_own_share']].drop_duplicates().reset_index(drop=True)

landown_new = pd.DataFrame()

for (h, f, v), group in landown.groupby(['HHID_dt', 'farming_output_code', 'visit_no']):
    row = {'HHID_dt': h, 'farming_output_code': f, 'visit_no': v}
    for _, entry in group.iterrows():
        row[entry['land_own']] = entry['land_own_share']
    landown_new = pd.concat([landown_new, pd.DataFrame([row])], ignore_index=True)
landown_new=landown_new.fillna(0)
landown_new=landown_new.rename(columns={ 'owned and possessed':'share_own_and_possessed',
                                         'leased in':'share_leased_in', 
                                         'otherwise possessed':'share_misc_possessed'})

landown_new.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/landownershipshares_vFeb2025.csv",index=False)

print('Farms that are irrigated = '+ str((landdf.if_land_irrigated==1).sum()))
print('Farms that are rainfed = '+str((landdf.if_land_irrigated==2).sum()))

'''
Farms that are irrigated = 49121
Farms that are rainfed = 21829
'''
irrigated=landdf[landdf.if_land_irrigated==1][['HHID_dt','visit_no','farming_output_code','primary_irrigation','second_irrigation']]
irrigated=pd.get_dummies(irrigated, columns=['primary_irrigation','second_irrigation'], dtype=int)
irrigated=irrigated.drop_duplicates().reset_index(drop=True)
irrigated=irrigated.groupby(['HHID_dt','visit_no','farming_output_code']).max().reset_index()
rainfed=landdf[landdf.if_land_irrigated==2][['HHID_dt','visit_no','farming_output_code']]
irrigated['if_land_irrigated']=1
rainfed['if_land_rainfed']=1
landdfT=pd.concat([irrigated,rainfed],axis=0).reset_index(drop=True)
landdfT=landdfT.fillna(0)
landdfT=landdfT.groupby(['HHID_dt','visit_no','farming_output_code']).max().reset_index()
landdfT['all_water_sources']=landdfT[['if_land_irrigated','if_land_rainfed']].sum(axis=1)
'''
landdfT.columns
Out[33]: 
Index(['HHID_dt', 'visit_no', 'farming_output_code', 'land_own_leased in',
       'land_own_otherwise possessed', 'land_own_owned and possessed',
       'primary_irrigation_canal', 'primary_irrigation_gw',
       'primary_irrigation_mixed', 'primary_irrigation_msw',
       'primary_irrigation_others', 'second_irrigation_canal',
       'second_irrigation_gw', 'second_irrigation_mixed',
       'second_irrigation_msw', 'second_irrigation_only_one_source',
       'second_irrigation_others', 'if_land_irrigated'],
      dtype='object')
'''
print('Shape of land ownership data = ',landown_new.shape)
print('Shape of land irrigation data = ',landdfT.shape)

landdf1=pd.merge(landown_new,landdfT,on=['HHID_dt','visit_no','farming_output_code'])
print('Shape of land ownership and land irrigation data = ',landdf1.shape)

print('The number of households in both = ',len(set(landdf1.HHID_dt).intersection(set(outputTdf.HHID_dt))))
# The number of households in both =  38451

print('outputTdf.shape = ',outputTdf.shape[0])
print('landdf1.shape = ',landdf1[['HHID_dt','farming_output_code','visit_no']].drop_duplicates().shape[0])
'''
outputTdf.shape =  77893
landdf1.shape =  64927
'''
if (landdf1[['HHID_dt','farming_output_code','visit_no']].drop_duplicates().shape[0]==landdf1.shape[0]):
    print('There are all unique points in landdf')
else:
    print('Fucking duplicates, mannn!')

#dropped_rows = test_left[test_left['share_leased_in'].isna()][['HHID_dt']]
#test=landdf1[landdf1['HHID_dt'].isin(dropped_rows)] ---> there are 0 rows in this df
test = pd.merge(outputTdf, landdf1, on=['HHID_dt', 'farming_output_code', 'visit_no'], how='inner') ## tested left merge, all dropped rows are HHID_dt without landdf1 information
print('The final shape of land ownership, irrigation and output = ', test.shape[0])
'''
The final shape of land ownership, irrigation and output =  62814

test.columns
Out[136]: 
Index(['HHID_dt', 'type', 'nss_district_code19', 'other_disposal_sale_Rs',
       'crop_code', 'unit', 'output_crop_rank', 'total_output_sale_kg',
       'all_disposal_sale_Rs', 'pre_harvest_value_Rs', 'harvest_value_Rs',
       'value_by_products_Rs', 'total_value_Rs', 'irrigated_acre',
       'output_irrigated_acre', 'unirrigated_acre', 'output_unirrigated_acre',
       'total_output_quantity', 'preharvest_acre', 'major_buyer',
       'sale_satistifaction', 'major_quantity_sold', 'major_sale_Rs.',
       'other_quantity_sold', 'farming_output_code', 'stock_output_sale_kg',
       'output_nan', 'visit_no', 'share_own_and_possessed', 'share_leased_in',
       'share_misc_possessed', 'primary_irrigation_canal',
       'primary_irrigation_gw', 'primary_irrigation_mixed',
       'primary_irrigation_msw', 'primary_irrigation_others',
       'second_irrigation_canal', 'second_irrigation_gw',
       'second_irrigation_mixed', 'second_irrigation_msw',
       'second_irrigation_only_one_source', 'second_irrigation_others',
       'if_land_irrigated', 'if_land_rainfed', 'all_water_sources']
'''
test.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/ownership_output_and_irrigation_Feb25.csv",index=False)
#%% Situation of Agriculture July 2018 to June 2019 -- HOUSEHOLD DEMOGRAPHICS

demov1df,demov1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L3B4demo_other_sample_hh.sav")
democols={'Common Primary Key for household identification':'HHID', 'Sector':'type', 
          'State':'state', 'State_District':'nss_district_code19',
          'Household size':'household_size','Religion code':'religion_code',
          'Social group code':'caste_code', 'Household classification  code':'household_type2',
       'usual consumer expenditure in a month for household purposes out of purchase (A)':'usual_consumption_purchase',
       'imputed value of usual consumption in a month from home grown stock (B)':'usual_consumption_home',
       'imputed value of usual consumption in a month from wages in kind, free collection, gifts, etc (C)':'usual_consumption_gifts',
       'expenditure on purchase of household durable during last 365 days (D)':'durable_purchase_last365days',
       'usual monthly consumer expenditure E: [A+B+C+(D/12)]':'household_cons_exp_monthly',
       'value of agricultural production from self-employment activities during the last 365 days(code)':'household_type1',
       'dwelling unit code':'if_owned_house', 'type of structure':'kaccha_or_pucca',
       'whether any of the household member has bank account':'if_bank_account',
       'whether any of the household member possesses MGNREG job card':'if_mnrega_card',
       'Whether undertook any work under MGNREG during the last 365 days':'if_mnrega_work_last365days',
       'Whether any of the household member is a member of registered farmers’ organisation':'if_farmer_organisation',
       'Whether the household possesses any Kisan Credit Card':'if_kisan_credit_card',
       'Whether the household possess Soil Health Card':'if_soil_health_card',
       'whether fertilizer, etc. applied to field as per recommendations of Soil Health Card':'compliance_soil_health_card',
       'whether the household possess Animal Health Card (Nakul Swasthya Patra':'if_animal_health_card',
       'whether the household insured any crop under PM Fasal Bima Yojana during last 365 days':'if_PM_fasal_bima_last365days',
       }
demov1df=demov1df[list(democols.keys())].rename(columns=democols)
demov1df['HHID']=demov1df['HHID'].astype(int)//10 ##removing visit_no since all visit_no=1
demov1df['HHID_dt']=(demov1df['HHID'].astype(float)*10000)+demov1df['nss_district_code19'].astype(float)
demov1df['HHID_dt']=demov1df['HHID_dt'].astype(str).apply(lambda x: x.split('.')[0])
demov1df.to_csv('G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/demohouseholds-vNov.csv')

# ***** ----- combining output and demo information ----- ******

outputdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/ownership_output_and_irrigation_Feb25.csv")
demodf=pd.read_csv('G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/demohouseholds-vNov.csv')
print('Number of households with output information = ',outputdf.HHID_dt.nunique())
print('Number of households with output and demographic information =',len(set(outputdf.HHID_dt).intersection(set(demodf.HHID_dt))))
'''
Number of households with output information =  37615
Number of households with output and demographic information = 37615
'''
test=pd.merge(outputdf,demodf,on=['HHID_dt', 'type', 'nss_district_code19'])
if test.shape[0]==outputdf.shape[0]:
    print('All households with output information have demographic information')
else:
    print('Some households are missing either information')
'''
OUT=> All households with output information have demographic information
'''

test.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/ownership_output_demo_irrigation_Feb25.csv",index=False)

#%% Situation of Agriculture July 2018 to June 2019 -- COST OF INPUTS

#we want to see if the cost of inputs are higher in the districts with drought than without
#we want to combinie the cost data with the output data to adjust the cost with the output of each household for the concerned crops
#to make prices comparable across the country, we adjust these values by the farmer CPI data

'''
Crop specific information, ie, crop_code, is only available for seeds, so we cannot filter by crop_code!=''
This is not the case for output where crop_code is left '' only for 'other' crops and for missing data

Filter input costs by the explicit_input_cost_Rs. column since this is the one completely filled out in the survey, 
and nan or '' values all indicate missing data
'''

costv1df,costv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L8B7input_expense_July_Dec_2018.sav")
costcols={'Common Primary Key for household identification':'HHID','Serial no.':'input_code',
          'Inputs-from where procured(Code)':'input_source_code','Inputs-quality/adequacy code':'input_quality_code',
          'Crop code -(as in col. 2 block 6)':'crop_code','State_District':'nss_district_code19',
          'Inputs-paid out expenses(Rs.)':'explicit_input_cost_Rs.',
          'Inputs-imputed expenses(Rs.)':'implicit_input_cost_Rs.'}
costv1df=costv1df[costcols.keys()].rename(columns=costcols)
costv1df=costv1df.dropna(subset='explicit_input_cost_Rs.')
#costv1df=costv1df[costv1df['crop_code']!='']
costv1df['crop_code']=costv1df['crop_code'].replace('',np.nan)
costv1df['crop_code']=costv1df['crop_code'].astype(float)
costv1df['HHID']=costv1df['HHID'].astype(float)
costv1df['input_code']=costv1df['input_code'].replace('',np.nan)
costv1df['input_code']=costv1df['input_code'].astype(float)
costv1df['input_source_code']=costv1df['input_source_code'].replace('',np.nan)
costv1df['input_source_code']=costv1df['input_source_code'].astype(float)
costv1df['input_quality_code']=costv1df['input_quality_code'].replace('',np.nan)
costv1df['input_quality_code']=costv1df['input_quality_code'].astype(float)
costv1df['nss_district_code19']=costv1df['nss_district_code19'].astype(int)
costv1df['HHID_dt']=(costv1df['HHID']*10000)+costv1df['nss_district_code19']
costv1df['implicit_input_cost_Rs.']=costv1df['implicit_input_cost_Rs.'].fillna(0)
costv1df['total_cost_Rs.']=costv1df['explicit_input_cost_Rs.']+costv1df['implicit_input_cost_Rs.']

costv2df,costv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V2L8B7input_expense_Jan_Jun_2019.sav")
costcols={'Common Primary Key for household identification':'HHID','Serial no.':'input_code',
          'Inputs-from where procured(code)':'input_source_code','Inputs-quality/adequacy code':'input_quality_code',
          'Crop code -(as in col. 2 block 6)':'crop_code','State_District':'nss_district_code19',
          'Inputs-paid out expenses(Rs.)':'explicit_input_cost_Rs.',
          'Inputs-imputed expenses(Rs.)':'implicit_input_cost_Rs.'}
costv2df=costv2df[costcols.keys()].rename(columns=costcols)
costv2df=costv2df.dropna(subset='explicit_input_cost_Rs.')
#costv2df=costv2df[costv2df['crop_code']!='']
costv2df['crop_code']=costv2df['crop_code'].replace('',np.nan)
costv2df['crop_code']=costv2df['crop_code'].astype(float)
costv2df['HHID']=costv2df['HHID'].astype(float)
costv2df['input_code']=costv2df['input_code'].replace('',np.nan)
costv2df['input_code']=costv2df['input_code'].astype(float)
costv2df['input_source_code']=costv2df['input_source_code'].replace('',np.nan)
costv2df['input_source_code']=costv2df['input_source_code'].astype(float)
costv2df['input_quality_code']=costv2df['input_quality_code'].replace('',np.nan)
costv2df['input_quality_code']=costv2df['input_quality_code'].astype(float)
costv2df['nss_district_code19']=costv2df['nss_district_code19'].astype(int)
costv2df['HHID_dt']=(costv2df['HHID']*10000)+costv2df['nss_district_code19']
costv2df['implicit_input_cost_Rs.']=costv2df['implicit_input_cost_Rs.'].fillna(0)
costv2df['total_cost_Rs.']=costv2df['explicit_input_cost_Rs.']+costv2df['implicit_input_cost_Rs.']
costv1df['visit_no']=costv1df['HHID_dt'].apply(lambda x: (str(x).split('.')[0])[-5])
costv2df['visit_no']=costv2df['HHID_dt'].apply(lambda x: (str(x).split('.')[0])[-5])

costTdf=pd.concat([costv1df,costv2df],axis=0)
#costTdf=costTdf[costTdf['crop_code'].isin(list(cropdict.keys()))] ##costTdf.shape = (77820,8)
costTdf['HHID_dt']=costTdf['HHID_dt'].apply(lambda x: str(x).split('.')[0])
costTdf['HHID_dt']=costTdf['HHID_dt'].apply(lambda x: x[:-5]+x[-4:])
## costTdf.shape = (511366,11)
costTdf.to_csv('G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/full_costdfv1v2.csv',index=False)


costTdf=pd.read_csv('G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/full_costdfv1v2.csv')
costTdf=costTdf.sort_values(by=['nss_district_code19','HHID_dt','visit_no','crop_code']).reset_index(drop=True)
costTdf=costTdf[['nss_district_code19','HHID_dt','visit_no','crop_code',
                 'input_code', 'input_source_code', 'input_quality_code',
                 'explicit_input_cost_Rs.','implicit_input_cost_Rs.','total_cost_Rs.']]
outputTdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/ownership_output_demo_irrigation_Feb25.csv") ##outputTdf.shape = (74667,27)
print('Cost df shape = ',costTdf.shape)
print('Ouptut df shape = ', outputTdf.shape)
'''
Cost df shape =  (484114, 11)
Ouptut df shape =  (62814, 69)
'''
print('The number of unique (HHID_dt, visit_no, crop_code) is = ',outputTdf[['HHID_dt','visit_no','crop_code']].drop_duplicates().shape[0])
if outputTdf[['HHID_dt','visit_no','crop_code']].drop_duplicates().shape[0] == outputTdf.shape[0]:
    print('There are no repeated crops cultivated by a household in each visit')
else:
    print('Problem= Household is reporting same crop cultivated multiple times in same visit')

## here need to adjust the cost and output dataframes before they can be combined
## then, for each household and crop-code, I can make a dataframe that has the 
## corresponding information about the inputs and their cost


costTdf=costTdf[costTdf.HHID_dt.isin(outputTdf.HHID_dt.unique())]
seeds=costTdf[costTdf['input_code']<=5]
nonseeds=costTdf[costTdf['input_code']>6]
nonseeds['input_name']=nonseeds['input_code'].apply(lambda x: inputdict[x]) 
nonseeds=nonseeds[['HHID_dt', 'visit_no','input_name','total_cost_Rs.']]
nonseeds= nonseeds.pivot(index=['HHID_dt', 'visit_no'], columns='input_name', values='total_cost_Rs.')
nonseeds=nonseeds.fillna(0).reset_index()
test=pd.merge(outputTdf,nonseeds,on=['HHID_dt','visit_no'],how='left')

#seeds information we can ignore source of procurement and quality

seeds=seeds[['HHID_dt','crop_code','visit_no','total_cost_Rs.']].groupby(['HHID_dt','crop_code','visit_no']).sum().reset_index().rename(columns={'total_cost_Rs.':'cost_seeds'})
test=pd.merge(test,seeds,on=['HHID_dt','visit_no','crop_code'],how='left')

'''
test['unit'].unique()
Out[209]: array([1], dtype=int64)

test.type.unique()
Out[210]: array([1], dtype=int64)

test.household_type1.unique()
Out[212]: array([2], dtype=int64)
'''
test=test[['state','nss_district_code19','HHID_dt', 'visit_no','farming_output_code','crop_code',
       'irrigated_acre','output_irrigated_acre', 'unirrigated_acre', 
       'output_unirrigated_acre','total_output_quantity', 
       'share_own_and_possessed', 'share_leased_in',
       'share_misc_possessed', 'primary_irrigation_canal',
       'primary_irrigation_gw', 'primary_irrigation_mixed',
       'primary_irrigation_msw', 'primary_irrigation_others',
       'second_irrigation_canal', 'second_irrigation_gw',
       'second_irrigation_mixed', 'second_irrigation_msw',
       'second_irrigation_only_one_source', 'second_irrigation_others',
       'if_land_irrigated', 'if_land_rainfed', 'all_water_sources',
       'cost_bio_fertlizer', 'cost_bio_pesticides', 'cost_chemical_pesticides','cost_chemical_fertilizer',
       'cost_crop_insurance', 'cost_crop_machinery', 'cost_diesel',
       'cost_electricity', 'cost_interest_on_crop_loan', 'cost_irrigation',
       'cost_labour_animal', 'cost_labour_human', 'cost_land_lease','cost_manure',
       'cost_minor_machine_repairs', 'cost_other', 'cost_total','cost_seeds',
       'household_size', 'religion_code','caste_code', 'household_type2','household_cons_exp_monthly',
       'if_owned_house', 'kaccha_or_pucca','if_bank_account', 'if_mnrega_card', 
       'if_mnrega_work_last365days','if_farmer_organisation', 'if_kisan_credit_card', 
       'if_soil_health_card','compliance_soil_health_card', 'if_animal_health_card',
       'if_PM_fasal_bima_last365days']]

spei4mo=pd.read_csv("G://SME/Paper_2/IndiaDrought/surveydata/situationofagriculture2019/spei4Mo_nss.csv")
visitdict={'201807-201812':1,'201901-201906':2,'201807-201906':0}
spei4mo['visit_no']=spei4mo['survey_reference'].apply(lambda x: visitdict[x])
print(spei4mo.columns)
'''
['survey_reference', 'severe_dry_moSPEI-4MO', 'extreme_dry_moSPEI-4MO',
       'exceptional_dry_moSPEI-4MO', 'mean_speiSPEI-4MO',
       'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 'min_speiSPEI-4MO',
       'std_speiSPEI-4MO', 'spei_district_code', 'nss_district_code19',
       'district_name', 'spei_state_name', 'visit_no']
'''
iddata=test[['nss_district_code19','visit_no']].drop_duplicates()
print('Number of districts with spei data = ',spei4mo.nss_district_code19.nunique())
iddata=pd.merge(iddata,spei4mo, how='left',on=['nss_district_code19','visit_no'])
print('Districts with nss data but without spei data: ')
print(iddata[iddata['mean_speiSPEI-4MO'].isna()][['nss_district_code19'
                                                 ]].drop_duplicates())
'''
      nss_district_code19
943                  2501 --- diu ---- no SPEI data ---- 24 observations in test
1104                 3101 - lakshwadeep -- no spei data - 4 observations in test
1193                 3404 - puducherry -- no spei data - 18 observations in test
1195                 3502 - andaman north and middle - no spei data - 31 obsvns

Total lost observations in nss data because no spei data = 77
'''
print('total observations in test with nss data = ',test.shape[0])
print('observations to be lost because no spei data = ',77)
test2=pd.merge(test,spei4mo,on= ['nss_district_code19', 'visit_no'],how='inner')
print('observations after merging = ',test2.shape[0])
if (test2.shape[0]+77==test.shape[0]):
    print('merging is fine, the columns are as follows')
    print(test2.columns)
    print('The number of observations with nss farm level data, for both visits, and the SPEI data = ', test2.shape[0])
else:
    print('unexpected losses')
'''
merging is fine, the columns are as follows
Index(['state', 'nss_district_code19', 'HHID_dt', 'visit_no',
       'farming_output_code', 'crop_code', 'irrigated_acre',
       'output_irrigated_acre', 'unirrigated_acre', 'output_unirrigated_acre',
       'total_output_quantity', 'share_own_and_possessed', 'share_leased_in',
       'share_misc_possessed', 'primary_irrigation_canal',
       'primary_irrigation_gw', 'primary_irrigation_mixed',
       'primary_irrigation_msw', 'primary_irrigation_others',
       'second_irrigation_canal', 'second_irrigation_gw',
       'second_irrigation_mixed', 'second_irrigation_msw',
       'second_irrigation_only_one_source', 'second_irrigation_others',
       'if_land_irrigated', 'if_land_rainfed', 'all_water_sources',
       'cost_bio_fertilizer', 'cost_bio_pesticides',
       'cost_chemical_pesticides', 'cost_crop_insurance',
       'cost_crop_machinery', 'cost_diesel', 'cost_electricity',
       'cost_interest_on_crop_loan', 'cost_irrigation', 'cost_labour_animal',
       'cost_labour_human', 'cost_land_lease', 'cost_manure',
       'cost_minor_machine_repairs', 'cost_other', 'cost_total', 'cost_seeds',
       'household_size', 'religion_code', 'caste_code', 'household_type2',
       'household_cons_exp_monthly', 'if_owned_house', 'kaccha_or_pucca',
       'if_bank_account', 'if_mnrega_card', 'if_mnrega_work_last365days',
       'if_farmer_organisation', 'if_kisan_credit_card', 'if_soil_health_card',
       'compliance_soil_health_card', 'if_animal_health_card',
       'if_PM_fasal_bima_last365days', 'survey_reference',
       'severe_dry_moSPEI-4MO', 'extreme_dry_moSPEI-4MO',
       'exceptional_dry_moSPEI-4MO', 'mean_speiSPEI-4MO',
       'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 'min_speiSPEI-4MO',
       'std_speiSPEI-4MO', 'spei_district_code', 'district_name',
       'spei_state_name'],
      dtype='object')
The number of observations with nss farm level data, for both visits, and the SPEI data =  62737
'''
## dealing with marginal farmers
test2['irrigated_acre']=test2['irrigated_acre'].fillna(0)
test2['unirrigated_acre']=test2['unirrigated_acre'].fillna(0)
test2['output_irrigated_acre']=test2['output_irrigated_acre'].fillna(0)
test2['output_unirrigated_acre']=test2['output_unirrigated_acre'].fillna(0)
test2.loc[(test2['irrigated_acre'] == 0) & (test2['output_irrigated_acre'] > 0), 'irrigated_acre'] = 0.0001
test2.loc[(test2['unirrigated_acre'] == 0) & (test2['output_unirrigated_acre'] > 0), 'unirrigated_acre'] = 0.0001

'''
        there are some issues with the irrigation indicator, so regenerating based
        on the irrigated_acre and unirrigated_acre values
'''
test2['if_land_irrigated']=(test2['irrigated_acre']>0).astype(int)
test2['if_land_rainfed']=(test2['unirrigated_acre']>0).astype(int)
filtered_df = test[(test['if_land_irrigated'] == 1) & ((test[['primary_irrigation_canal', 'primary_irrigation_gw', 
                                                              'primary_irrigation_mixed', 'primary_irrigation_msw', 
                                                              'primary_irrigation_others', 'second_irrigation_canal', 
                                                              'second_irrigation_gw', 'second_irrigation_mixed', 
                                                              'second_irrigation_msw', 'second_irrigation_only_one_source', 
                                                              'second_irrigation_others']] == 0)).all(axis=1)]
if filtered_df.shape[0]==0:
    print('If the land is irrigated, then the method of irrigation is always mentioned.')

filtered_df = test[(test['if_land_irrigated'] == 0) & ((test[['primary_irrigation_canal', 'primary_irrigation_gw', 
                                                              'primary_irrigation_mixed', 'primary_irrigation_msw', 
                                                              'primary_irrigation_others', 'second_irrigation_canal', 
                                                              'second_irrigation_gw', 'second_irrigation_mixed', 
                                                              'second_irrigation_msw', 'second_irrigation_only_one_source', 
                                                              'second_irrigation_others']] == 1)).any(axis=1)]   
if filtered_df.shape[0]==0:
    print('There is no irrigation methodology mentioned if land is not irrigated.')
    
test2['all_water_sources']=test2[['if_land_irrigated','if_land_rainfed']].sum(axis=1)
test2=test2[test2['all_water_sources']>0]
test2.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/output_water_cost_demo_spei_vFeb25.csv",index=False)

## ******** ##
test2.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/output_water_cost_demo_spei_vFeb25.csv",index=False)
## ******** ##


## also do it with spei12MO eventually
#%% Situation of Agriculture July 2018 to June 2019 -- CROP LOSS DATA

lossv1df,lossv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L18B16_crop_loss.sav")
losscols={'Common Primary Key for household identification':'HHID','State_District':'nss_district_code19',
          'Visit number':'visit_no', 'Crop code as in col.2 of block 6':'crop_code',
          'Is the Crop insured?':'is_crop_insured','any  crop loss':'if_crop_loss', 
          'cause of crop loss':'cause_crop_loss'}
lossv1df=lossv1df[list(losscols.keys())].rename(columns=losscols)   
lossv1df['HHID']=lossv1df['HHID'].apply(lambda x: x[:-1])
lossv1df['HHID_dt']=lossv1df['HHID']+lossv1df['nss_district_code19']

lossv2df,lossv2meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V2L18B16_crop_loss.sav")
lossv2df=lossv2df[list(losscols.keys())].rename(columns=losscols)   
lossv2df['HHID']=lossv2df['HHID'].apply(lambda x: x[:-1])
lossv2df['HHID_dt']=lossv2df['HHID']+lossv2df['nss_district_code19']

lossdf=pd.concat([lossv1df,lossv2df],axis=0)
lossdf['crop_code']=lossdf['crop_code'].astype(int)
lossdf=lossdf[lossdf['crop_code'].isin(list(cropdict.keys()))]
lossdf=lossdf.drop('HHID',axis=1)
lossdf['nss_district_code19']=lossdf['nss_district_code19'].astype(float)
lossdf['visit_no']=lossdf['visit_no'].astype(int)
#lossdf.shape
#Out[132]: (77711, 8)
lossdf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/croplossv1v2.csv",index=False)
## combine crop loss data with climate data and demographics data separately since there are are some observations 
## in lossdf that are being dropssed in the finaldf since some households reported no agricultural
## output or further details.
landdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/landofhouseholdsv1v2-vNov.csv")
#landdf.shape
# 77534,12
lossdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/croplossv1v2.csv")
#lossdf.shape
# 77711,7
demodf=pd.read_csv('G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/demohouseholds-vNov.csv')
demodf=demodf.drop('Unnamed: 0',axis=1)
speidata=pd.merge(drought,nssspei,on='spei_district_code') ##675 districts have matched nss and spei codes
speidata=speidata.rename(columns={'nss_district_code':'nss_district_code19'})
## pd.merge(drought,nssspei,on='spei_district_code',how='outer')['spei_district_code'].unique().shape
## Out[23]: (734,)
speidata=speidata[['spei_district_code','nss_district_code19','nss_state_name','district_name',
                   'visit_no', 'mean_spei', 'median_spei', 'max_spei',
       'min_spei', 'std_spei']].rename(columns={'nss_state_name':'state_name'})

df1=pd.merge(demodf,lossdf,on=['nss_district_code19','HHID_dt'])
df1=pd.merge(df1,landdf,on=['HHID_dt','visit_no']) ## shape=(77104,41)
df1=pd.merge(df1,speidata,on=['nss_district_code19','visit_no']) ## shape = (76829,49)
## there are 666 unique districts in lossdf, 668 unique districts in demodf
df1.to_csv('G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/croplossdf.csv',index=False)

#%% Expense on productive Assets -- asset_code 1 to 10 is for farm business

# asset codes 03, 04 and 05 are for livestock, so we can ignore

assetdict={'01':'asset_land_EXPRs','02':'asset_building_EXPRs','03':'asset_fish_EXPRs','04':'asset_livestock_EXPRs',
           '05':'asset_poultryEXPRs','06':'asset_farm_eqp_1_EXPRs','07':'asset_farm_eqp2_EXPRs','08':'asset_farm_eqp3_EXPRs',
           '09':'asset_water_eqpEXPRs','10':'asset_others_EXPRs'} ## farm_eqp1 = basic equipments, farm_eqp2 = electronic equipments, we only consider the assets used in farm business

assetv1df,assetv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L14B12productive_assets_Jul_Dec_2018.sav")
assetcols={'Common Primary Key for household identification':'HHID','Serial no.':'asset_code','State_District':'nss_district_code19',
          'Expenditure incurred-Purchase(Rs.)':'asset_purchase_cost_Rs',
          'Expenditure incurred-Major repair(Rs.)':'asset_repair_cost_Rs',
          'Receipt  from sale(Rs.)':'asset_sale_receipt_Rs',
          'net expenditure  (col 3+4-5)(Rs.)':'asset_net_payoff_Rs'}
assetv1df=assetv1df.rename(columns=assetcols)
assetv1df['asset_name']=assetv1df['asset_code'].apply(lambda x: assetdict[x])
assetv1df['asset_code']=assetv1df['asset_code'].astype(int)
assetv1df['HHID_dt']=(assetv1df['HHID'].astype(float)*10000)+assetv1df['nss_district_code19'].astype(float)
assetv1df['HHID_dt'] = assetv1df['HHID_dt'].apply(lambda x: f'{x:.0f}')
assetv1df['HHID_dt'] = assetv1df['HHID_dt'].astype(str) 
assetv1df['visit_no']=assetv1df['HHID_dt'].apply(lambda x: str(x)[-5])
assetv1df['HHID_dt']=assetv1df['HHID_dt'].apply(lambda x: str(x)[:-5]+str(x)[-4:])
assetv1df=assetv1df[['HHID_dt','asset_name','asset_net_payoff']]

#%% making CGDE OLS dataframes for rice and wheat
'''
## ADD CHEMICAL FERTILIZER
costdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/full_costdfv1v2.csv")
costdf=costdf[costdf['input_code']==6][['HHID_dt','visit_no','total_cost_Rs.']]
costdf['cost_chemical_fertilizer']=costdf['total_cost_Rs.'].fillna(0)
costdf=costdf[['HHID_dt','visit_no','cost_chemical_fertilizer']]
costdf=costdf.groupby(['HHID_dt','visit_no']).sum().reset_index() ## the same number of observations
Tdf=pd.merge(Tdf,costdf,on=['HHID_dt','visit_no'],how='left')
Tdf['cost_chemical_fertilizer']=Tdf['cost_chemical_fertilizer'].fillna(0)
Tdf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/output_water_cost_demo_spei_vFeb25.csv",index=False)
'''

## adding crop market rate

outputv1df,outputv1meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V1L7B6output_crops_July_Dec_2018.sav")
outputv2df,outputv2meta=makedata("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/V2L7B6output_crops_Jan_Jun_2019.sav")
coldict={'Common Primary Key for household identification':'HHID',
        'State_District':'nss_district_code19','Crop code':'crop_code',
        'Visit number':'visit_no','Rate (Rs.0.00): col.17/col.16':'crop_market_rate_Rs'}
outputv1df=outputv1df[coldict.keys()].rename(columns=coldict)
outputv2df=outputv2df[coldict.keys()].rename(columns=coldict)
outputv1df['HHID']=outputv1df['HHID'].astype(float)//10
outputv1df['HHID_dt']=(outputv1df['HHID'].astype(float)*10000)+outputv1df['nss_district_code19'].astype(float)
outputv2df['HHID']=outputv2df['HHID'].astype(float)//10
outputv2df['HHID_dt']=(outputv2df['HHID'].astype(float)*10000)+outputv2df['nss_district_code19'].astype(float)
outputv1df['visit_no']=1
outputv2df['visit_no']=2
outputd=pd.concat([outputv1df,outputv2df],axis=0)[['nss_district_code19','HHID_dt','visit_no','crop_code','crop_market_rate_Rs']]
outputd['HHID_dt']=outputd['HHID_dt'].astype('int64')
outputd=outputd[outputd['crop_code']!='']
outputd['crop_code']=outputd['crop_code'].astype('int64')
Tdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/output_water_cost_demo_spei_vFeb25.csv") (62735,75)
outputd=outputd[(outputd['HHID_dt'].isin(Tdf.HHID_dt.unique()))&
                (outputd['crop_code'].isin(Tdf.crop_code.unique()))]
outputdf=outputd[['nss_district_code19','HHID_dt','visit_no','crop_code','crop_market_rate_Rs']].sort_values(by=['HHID_dt','visit_no','crop_code'])
outputdf['crop_district_rate_Rs']=outputdf[['nss_district_code19','visit_no','crop_code','crop_market_rate_Rs']].groupby(['nss_district_code19','crop_code','visit_no']).transform('mean')
outputdf=outputdf.sort_values(by=['crop_code','nss_district_code19','visit_no'])      
outputdf['crop_district_rate_Rs_1']=((outputdf['crop_market_rate_Rs'].isna()).astype(int))*outputdf['crop_district_rate_Rs']                                                                                                                      
outputdf['crop_market_rate_Rs']=outputdf['crop_market_rate_Rs'].fillna(0)
outputdf['crop_market_rate_Rs']=outputdf['crop_market_rate_Rs']+outputdf['crop_district_rate_Rs_1']
outputdf=outputdf[['HHID_dt','visit_no','crop_code','crop_market_rate_Rs']]
Tdf=pd.merge(Tdf,outputdf,on=['HHID_dt','visit_no','crop_code']) ## (62735,75)
Tdf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/output_water_cost_demo_spei_vFeb25.csv",index=False)

Tdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/output_water_cost_demo_spei_vFeb25.csv") ##62735 rows, 74 columns
Tdf['crop_name']=Tdf['crop_code'].apply(lambda x: cropdict[x])
Tdf['mixed']=Tdf['if_land_irrigated']+Tdf['if_land_rainfed'] ## 1765 are mixed, we remove them
Tdf=Tdf[Tdf['mixed']==1] ## 60970 observations
# checking if weird irrigated_acre=0 and output_irrigated_acre>0
if ((Tdf['output_irrigated_acre']>0)&(Tdf['irrigated_acre']==0)).sum()==0:
    print('No farms with irrigated_acre=0 but output_irrigated_acre>0')
else:
    print('There are farms with 0 irrigated acre but positive output from irrigated acre!')
if ((Tdf['output_unirrigated_acre']>0)&(Tdf['unirrigated_acre']==0)).sum()==0:
    print('No farms with unirrigated_acre=0 but output_unirrigated_acre>0')
else:
    print('There are farms with 0 unirrigated acre but positive output from unirrigated acre!')
'''
Tdf['mixed']=Tdf[['if_land_irrigated','if_land_rainfed']].sum(axis=1)

Tdf['mixed'].unique()
Out[14]: array([1], dtype=int64)
Tdf[Tdf['unirrigated_acre'].isna()].irrigated_acre.sum()
Out[17]: 0.0 ##--- >> No nan acre

Tdf[Tdf['unirrigated_acre'].isna()].unirrigated_acre.sum()
Out[18]: 0.0 ##--- >> No nan acre
'''
Tdf['farm_output_kg']=Tdf[['output_irrigated_acre','output_unirrigated_acre']].sum(axis=1)
Tdf['farm_land_m2']=(Tdf[['irrigated_acre','unirrigated_acre']].sum(axis=1))*4046.86
Tdf['total_cost']=Tdf[[ 'cost_bio_fertilizer', 'cost_bio_pesticides',
 'cost_chemical_pesticides','cost_chemical_fertilizer',
 'cost_irrigation', 'cost_labour_human','cost_manure','cost_seeds']].fillna(0).sum(axis=1)
q25 = Tdf["household_cons_exp_monthly"].quantile(0.25)
q50 = Tdf["household_cons_exp_monthly"].quantile(0.50)
q75 = Tdf["household_cons_exp_monthly"].quantile(0.75)
Tdf["hh_expense_grp"] = pd.cut(
    Tdf["household_cons_exp_monthly"],
    bins=[-float("inf"), q25, q50, q75, float("inf")],  # Ensure all values are included
    labels=[1, 2, 3, 4],  # Assigning category labels
    include_lowest=True)
Tdf["hh_expense_grp"] = Tdf["hh_expense_grp"].astype(int)
Tdf['is_non_backward_caste']=(Tdf['caste_code']==9).astype(int) ## if caste_code =9, then it is a non-backward caste
'''
Tdf['is_non_backward_caste'].unique()
Out[55]: array([1, 0])

Tdf['hh_expense_grp'].unique()
Out[56]: array([4, 1, 3, 2])

Then hh demo groups are (1,1),(1,2),(1,3),(1,4),
                        (0,1),(0,2),(0,3),(0,4)
'''
Tdf["hh_demo_grp"] = Tdf["is_non_backward_caste"].astype(str) + Tdf["hh_expense_grp"].astype(str)
group_mapping = {
    "11": 1, "12": 2, "13": 3, "14": 4,  # Non-backward caste groups
    "01": 5, "02": 6, "03": 7, "04": 8   # Backward caste groups
}
Tdf["hh_demo_grp"] = Tdf["hh_demo_grp"].map(group_mapping)
Tdf["hh_demo_grp"] = Tdf["hh_demo_grp"].astype(int)
Tdf[["is_non_backward_caste", "hh_expense_grp", "hh_demo_grp"]].head(20)

for col in ['if_farmer_organisation', 'if_kisan_credit_card', 'if_soil_health_card']:
    Tdf[col]=(Tdf[col]==1).astype(int)

Tdf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/regdf_vFeb25.csv") #(60970,83)
'''
Tdf.columns
['state','visit_no','mean_speiSPEI-4MO',hh_demo_grp,'crop_name',
 'farm_output_kg','farm_land_m2','if_land_irrigated', 'total_cost',
 ]
'''
Tdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/regdf_vFeb25.csv")
Tdf['total_cost_sum']=Tdf[['cost_bio_fertilizer',
 'cost_bio_pesticides',
 'cost_chemical_pesticides',
 'cost_crop_insurance',
 'cost_crop_machinery',
 'cost_diesel',
 'cost_electricity',
 'cost_interest_on_crop_loan',
 'cost_irrigation',
 'cost_labour_animal',
 'cost_labour_human',
 'cost_land_lease',
 'cost_manure',
 'cost_minor_machine_repairs',
 'cost_other',
 'cost_seeds',
 'cost_chemical_fertilizer']].sum(axis=1)
print(Tdf.columns)
tempdf=pd.read_csv('G://SME/paper_2/IndiaDrought/speidata/sitagri2019_t2m.csv')
Tdf=pd.merge(Tdf,tempdf,on=['spei_district_code','visit_no'])## for the CGDE workshop, then find temperature data for speicodes=[104, 171, 351, 601]

era5=pd.read_csv("G://SME/paper_2/IndiaDrought/era5_data/spei_districts_era5.csv")
Tdf_kl=pd.merge(Tdf,era5,on=['spei_district_code','visit_no'])
Tdf_kl.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/regdf_vMar25.csv",index=False) ## <<-- MAIN LINK TO DATA USED FOR MODELS

Tdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/regdf_v2Mar25.csv")
ricedf=Tdf[Tdf['crop_name']=='paddy'].drop(['crop_code','crop_name',
                                            'nss_district_code19',
                                            'HHID_dt','farm_output_kg',
                                            'spei_district_code'],axis=1)
ricedf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/ricedf_vMar25.csv",index=False)

Tdf=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/regdf_v2Mar25.csv")
ricedf1=Tdf[(Tdf['crop_name']=='paddy')&(Tdf['visit_no']==1)] ##shape == nunique(HHID_dt) = 22332
ricedf1["ln_farm_crop_yield"]=np.log(ricedf1["farm_output_kg"]/ricedf1["farm_land_m2"])
ricedf1=ricedf1.drop(['crop_code','crop_name',
                      'nss_district_code19','state',
                      'HHID_dt','farm_output_kg',
                      'spei_district_code','visit_no'],axis=1)
ricedf1.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/ricedf_vMar25_onlyvisit1.csv",index=False)

wheatdf1=Tdf[(Tdf['crop_name']=='wheat')&(Tdf['visit_no']==2)] ##shape == nunique(HHID_dt) = 13595
wheatdf1["ln_farm_crop_yield"]=np.log(wheatdf1["farm_output_kg"]/wheatdf1["farm_land_m2"])
wheatdf1=wheatdf1.drop(['crop_code','crop_name',
                      'nss_district_code19','state',
                      'HHID_dt','farm_output_kg',
                      'spei_district_code','visit_no'],axis=1)
wheatdf1.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/wheatdf_vApr25_onlyvisit2.csv",index=False)

regdf=Tdf[['HHID_dt','spei_district_code','state','visit_no','mean_speiSPEI-4MO',
       'median_speiSPEI-4MO', 'max_speiSPEI-4MO', 'min_speiSPEI-4MO',
       'std_speiSPEI-4MO','crop_name','farm_output_kg','farm_land_m2',
       'crop_market_rate_Rs''if_land_irrigated','if_farmer_organisation', 
       'if_kisan_credit_card','if_soil_health_card','is_non_backward_caste',
       'total_cost_sum','share_own_and_possessed', 'share_leased_in',
       'share_misc_possessed', 'primary_irrigation_canal',
       'primary_irrigation_gw', 'primary_irrigation_mixed',
       'primary_irrigation_msw', 'primary_irrigation_others']]

regdf['crop_m2']=regdf[['spei_district_code','visit_no','crop_name','farm_land_m2']].groupby(['spei_district_code','visit_no','crop_name']).transform('sum')
regdf['hh_crop_m2']=regdf[['HHID_dt','spei_district_code','visit_no','crop_name','farm_land_m2']].groupby(['HHID_dt','spei_district_code','visit_no','crop_name']).transform('sum')
regdf['district_m2']=regdf[['spei_district_code','visit_no','farm_land_m2']].groupby(['spei_district_code','visit_no']).transform('sum')
regdf['hh_m2']=regdf[['HHID_dt','spei_district_code','visit_no','farm_land_m2']].groupby(['HHID_dt','spei_district_code','visit_no']).transform('sum')
regdf['share_crop_m2']=regdf['crop_m2']/regdf['district_m2']
regdf['hh_share_crop_m2']=regdf['hh_crop_m2']/regdf['hh_m2']
regdf['irrigated_m2']=regdf['if_land_irrigated']*regdf['farm_land_m2']
regdf['district_irrigated_m2']=regdf[['spei_district_code','visit_no','irrigated_m2']].groupby(['spei_district_code','visit_no']).transform('sum')
regdf['hh_irrigated_m2']=regdf[['HHID_dt','spei_district_code','visit_no','irrigated_m2']].groupby(['spei_district_code','HHID_dt','visit_no']).transform('sum')
regdf['hh_irrigated_share']=regdf['hh_irrigated_m2']/regdf['hh_m2']
regdf['share_irrigated_m2']=regdf['district_irrigated_m2']/regdf['district_m2']
regdf['farm_yield']=regdf['farm_output_kg']/regdf['farm_land_m2']
regdf['hh_farm_yield']=regdf[['HHID_dt','visit_no','farm_yield']].groupby(['HHID_dt','visit_no']).transform('mean')
regdf['farm_yield_Rs']=regdf['farm_yield']*regdf['crop_market_rate_Rs']
regdf['total_cost_sum_perm2']=regdf['total_cost_sum']/regdf['farm_land_m2']
regdf['hh_total_cost_sum_perm2']=regdf[['HHID_dt','visit_no','total_cost_sum_perm2']].groupby(['HHID_dt','visit_no']).transform('mean')
regdf['district_crop_kg']=regdf[['spei_district_code','visit_no','crop_name','farm_output_kg']].groupby(['spei_district_code','visit_no','crop_name']).transform('sum')
regdf['district_crop_yield']=regdf['district_crop_kg']/regdf['district_m2']
regdf['district_kg']=regdf[['spei_district_code','visit_no','farm_output_kg']].groupby(['spei_district_code','visit_no']).transform('sum')
regdf['district_yield']=regdf['district_kg']/regdf['district_m2']

regdfv1=regdf[regdf['visit_no']==1]
cropshare=regdfv1[['crop_name','hh_share_crop_m2','spei_district_code']].groupby(['spei_district_code','crop_name']).mean().reset_index()
cropshare=cropshare.pivot(columns='crop_name',values='hh_share_crop_m2',index='spei_district_code')
cropshare=cropshare.fillna(0).reset_index()

hhdf=regdfv1[['farm_yield','if_farmer_organisation','if_kisan_credit_card',
                     'if_soil_health_card','is_non_backward_caste','hh_irrigated_share',
                     'hh_farm_yield','hh_total_cost_sum_perm2','share_irrigated_m2',
                     'spei_district_code','mean_speiSPEI-4MO', 'std_speiSPEI-4MO','state']].groupby([ 'spei_district_code','mean_speiSPEI-4MO', 'std_speiSPEI-4MO','state']).mean().reset_index()

districtdf=pd.merge(hhdf,cropshare,on='spei_district_code')
districtdf=districtdf.fillna(0)

temp=pd.read_csv("G://SME/paper_2/IndiaDrought/regression_data/regression_df/districtOLS_95.csv")
temp=temp[['spei_district_code','mean_t2m','std_t2m']]
districtdf=pd.merge(districtdf,temp,on='spei_district_code')
districtdf.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/hhallcropsOLS_vMar.csv",index=False)

Tdf=pd.merge(Tdf,tempdf, on=['visit_no','spei_district_code'],how='left') ## Tdf saved with temperature data

for col in ['cost_bio_fertilizer', 'cost_bio_pesticides',
'cost_chemical_fertilizer','cost_chemical_pesticides',
'cost_crop_machinery','cost_irrigation','cost_labour_human',
'cost_seeds' , 'total_cost_sum']:
      regdf[col+'_per_m2']=regdf[col]/regdf['farm_land_m2']
      regdf[col+'_per_gm']=regdf[col]/(regdf['farm_output_kg']*1000)

def removeoutliers(df,varname):
    bounds={}
    for var in varname:
        bounds[var+'_low']=df[var].quantile(0.10)
        bounds[var+'_high']=df[var].quantile(0.90)
    df=df[(df[varname[0]]>=bounds[varname[0]+'_low']) &
          (df[varname[1]]>=bounds[varname[1]+'_low']) &
          (df[varname[0]]<=bounds[varname[0]+'_high']) &
          (df[varname[1]]<=bounds[varname[1]+'_high'])]
    return df

riceols=regdf[regdf['crop_name']=='paddy'].drop('crop_name',axis=1) #(26918, 7)
riceols=removeoutliers(riceols,['farm_yield_Rs','total_cost_sum_perm2'])


wheatols=regdf[regdf['crop_name']=='wheat'].drop('crop_name',axis=1)#(13759, 26)
wheatols=removeoutliers(wheatols,['farm_yield_Rs','total_cost_sum_perm2']) #(9309, 26)

riceols.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/riceols_vFeb25.csv",index=False)
wheatols.to_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/wheatols_vFeb25.csv",index=False)

## making district mean and std aggregation of rice OLS
std_spei=Tdf[['spei_district_code','std_speiSPEI-4MO','visit_no']].drop_duplicates()
riceols=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/riceols_vFeb25.csv")
riceols=pd.merge(riceols,tempdf,on=['spei_district_code','visit_no']) ### fix missing districts after CGDE
riceols=pd.merge(riceols,std_spei,on=['spei_district_code','visit_no'])
riceols.to_csv("G://SME/paper_2/IndiaDrought/regression_data/regression_df/cgde_riceols.csv",index=False)    

wheatols=pd.read_csv("G://SME/paper_2/IndiaDrought/surveydata/situationofagriculture2019/wheatols_vFeb25.csv")
wheatols=pd.merge(wheatols,tempdf,on=['spei_district_code','visit_no']) ### fix missing districts after CGDE
wheatols=pd.merge(wheatols,std_spei,on=['spei_district_code','visit_no'])
wheatols.to_csv("G://SME/paper_2/IndiaDrought/regression_data/regression_df/cgde_wheatols.csv",index=False)    

## next do wheat regressions and then organise the display of the regression results in CGDE slides

