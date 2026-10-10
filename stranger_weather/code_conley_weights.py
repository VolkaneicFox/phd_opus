import pandas as pd
import geopandas as gpd

shp_kreise=gpd.read_file("C://Users/nb/Desktop/paper_one/official_de_shapefiles.gk3.shape.ebenen/vg5000_ebenen_1231/VG5000_KRS.shp",crs=4326)
shp_kreise=shp_kreise[['ARS','NUTS','geometry']]
shp_kreise['centroid']=shp_kreise['geometry'].centroid
shp_kreise['centroid_deg']=shp_kreise.centroid.to_crs(epsg=4326)
shp_kreise['lat_deg']=shp_kreise.centroid_deg.y
shp_kreise['lon_deg']=shp_kreise.centroid_deg.x
distances=shp_kreise[['ARS','lon_deg','lat_deg']]
distances=distances.rename(columns={'ARS':'Kreise_Code'})
distances['Kreise_Code']=distances['Kreise_Code'].astype(int)
x1_rows=distances['lon_deg'].values.reshape(1,-1)
x2_rows=distances['lon_deg'].values.reshape(-1,1)
x2_minus_x1=x2_rows-x1_rows
y1_rows=distances['lat_deg'].values.reshape(1,-1)
y2_rows=distances['lat_deg'].values.reshape(-1,1)
y2_minus_y1=y2_rows-y1_rows
euc_deg=np.sqrt((x2_minus_x1**2)+(y2_minus_y1**2))
euc_dist=euc_deg*111

bound=250
euc_wts=np.maximum(1 - euc_dist / bound, 0)
euc_wts=pd.DataFrame(euc_wts.copy())
kreise_list=list(distances.Kreise_Code.astype(int))
euc_wts.index=kreise_list
euc_wts.columns=kreise_list
#euc_wts.to_csv("C:\\Users/nb/Desktop/paper_one/conley_weights.csv")