import yfinance as yfin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
drive_path=""
"""
The commodities futures are: coffee, sugar, corn and soybean
The economic controls, Brazilian Exchange Rate

Taking daily rate of change, removing the auto-regressive effect and the shock
of the exchange rate, that we believe covers most news and economy wide shock, 
the residual is then predicted against the climate variables.

Since Brazilian Rial Exchange Rate Data starts from 2004, take one section of 
futures prices that is from 2004-01-01 to 2025-12-31, with the residuals as the
price subject to climate shocks.

Suppose the price shock happens at t0, then the climate shock might have happened
in time t{0-k}, where k=0,1,2,3...T, for T not too large. Therefore, there must
be  an alignment between the  periods prior for  which the currency relation is 
considered and the climate shock timing.

Starting from 2003-12-01 (the first time-stamp on the BRL), take a few differe-
nt time  periods, and within  each period, split the series into training (80%) 
and test. This gives different period  specific models, then take all the bands
of  test  data and run them through all the  differently trained models, so for 
each point we get a sequence of predictions. 

The pipeline is two steps: 1. to orthogonalise on the currency and 2. to predi-
ct the log.diff on its own history and the climate shocks. 

For each commodity, the climate data needs to be selected for the regions grow-
ing the relevant crops.

Step 1. Period wise, make training and test data, and orthogonalise on the cur-
rency data

"""
commodities={"KC=F":"Arabica Futures",
             "ZS=F":"Soybean Futures",
             "SB=F":"Sugar No. 11",
             "ZC=F":"Corn Futures",
             "BRLUSD=X":"Rial to USD",
             #"BZ=F":"Brent Crude", using daily USFRED data
             "USDBOB=X":"Boliv to USD",
             "ARSUSD=X":"Arg  Peso to USD", ##RiP
             "PYG=X":"Parag Gua to USD"
             }
def gather_futures(ticker,start_date,end_date,price="Close"):
    finance=yfin.Ticker(ticker).history(start=start_date,
                                        end=end_date)
    pct_change= np.log(finance[[price]]).diff()
    pct_change.columns=[ticker]
    pct_change= pct_change.reset_index()
    pct_change["Date"]=pct_change["Date"].apply(lambda x: str(x).split(" ")[0].strip())
    pct_change=pct_change[["Date",ticker]].set_index("Date")
    return pct_change

def tickPrice(ticker,start_date,end_date,price="Close"):
    finance=yfin.Ticker(ticker).history(start=start_date,
                                        end=end_date)
    fdf = finance[[price]]
    fdf.columns = [f"price{ticker}"]
    fdf = fdf.reset_index()
    fdf["Date"]=fdf["Date"].apply(lambda x: str(x).split(" ")[0].strip())
    fdf=fdf[["Date",f"price{ticker}"]].set_index("Date")
    return fdf
    
for ticker in ["KC=F","ZS=F"]:
    test = tickPrice(ticker=ticker, start_date="1995-01-01", end_date="2026-01-01")
    test.to_csv(drive_path+f"/timeseries_csv/{ticker}_mainseries.csv")
    

def plot_future_logdiff(self):
    fix,axes=plt.subplots(nrows=len(commodities.keys()),ncols=1,figsize=(20,50))
    for index,item in enumerate(commodities):
        df=gather_futures(ticker=item, start_date="1995-01-01", end_date="2026-01-01")
        df.plot(ax=axes[index])
        axes[index].set_title(commodities[item])
    plt.show()

comdf_list=[]
for ticker in commodities:
    df=gather_futures(ticker=ticker, start_date="2000-01-01", end_date="2026-01-01")
    comdf_list.append(df)
    
comdf=pd.concat(comdf_list,axis=1)
comdf=comdf.sort_index()

brent=pd.read_csv(f"{drive_path}/timeseries_csv/brenUSFRED.csv")
brent=brent.set_index("observation_date")
brent.columns=["usdbrent"]
brent=np.log(brent).diff()
comdf=pd.concat([comdf,brent],axis=1)
comdf=comdf.sort_index()
comdf_sel=comdf.loc["2000-01-01":"2025-12-31"]
comdf_sel.to_csv(f"{drive_path}/timeseries_csv/commodities.csv")

rial_br=pd.read_csv(f"{drive_path}/timeseries_csv/brrialtousdusfred.csv") ##https://fred.stlouisfed.org/series/DEXBZUS
rial_br=rial_br.set_index("observation_date")
rial_br.columns=["rial_usd_br"]
rial_br=np.log(rial_br).diff()
rial_br=rial_br.loc["2000-01-01":"2025-12-31"]
commdf_sel=pd.read_csv(f"{drive_path}/timeseries_csv/commodities.csv")
commdf_sel=commdf_sel.set_index("Unnamed: 0")
commdf_sel=pd.concat([commdf_sel,rial_br],axis=1)
commdf_sel=commdf_sel.sort_index()
commdf_sel=commdf_sel.loc["2000-01-01":"2025-12-31"]
commdf_sel.to_csv(f"{drive_path}/timeseries_csv/commodities.csv")
