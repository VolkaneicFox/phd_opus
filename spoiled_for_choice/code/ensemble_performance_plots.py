import pandas as pd
import matplotlib.pyplot as plt

def make_plot_data(full_df):
    random_states=[3504,2992,3529,5042,4053,2492,9193,6843,3923,5935]
    sample_size=full_df.shape[0]//len(random_states)
    plot_dfs=[]
    for state_space in random_states:
        seldf=full_df[['y_obs',f'mea_{state_space}']].rename(columns={f'mea_{state_space}':'y_mea'})
        seldf=seldf.sample(n=sample_size)
        seldf['state_space']=state_space
        drop_indices=seldf.index
        full_df=full_df.drop(drop_indices,axis=0)
        plot_dfs.append(seldf)
    plot_df=pd.concat(plot_dfs,axis=0)
    
    random_colors={3504:'limegreen',2992:'cornflowerblue',3529:'blueviolet',5042:'indigo',
                   4053:'mediumturquoise',2492:'gold',9193:'darkorange',6843:'crimson',
                   3923:'indianred',5935:'chocolate'}
    plot_df['color']=plot_df['state_space'].map(random_colors)
    
    return plot_df

def plot_ensemble_predictions(plot_df):
    plot_df['res']=(plot_df['y_obs']-plot_df['y_mea'])**2
    fig,ax=plt.subplots(ncols=2,nrows=1,figsize=(20,6))
    ax[0].scatter(plot_df['y_obs'],plot_df['y_mea'],s=0.5,c=plot_df['color'])
    ax[0].plot(plot_df['y_obs'],plot_df['y_obs'],c='b',linewidth=0.1)
    ax[0].set_title('(a) Observed vs Predicted Yield',fontsize=16)
    ax[0].set_xlabel('Observed Farm Yields',fontsize=14)
    ax[0].set_ylabel('Predicted Farm Yields',fontsize=14)
    ax[1].scatter(plot_df['y_obs'],plot_df['res'],s=0.5,c=plot_df['color'])
    ax[1].set_title('(b) Observed Yield vs Squared-Residuals',fontsize=16)
    ax[1].set_xlabel('Observed Farm Yields',fontsize=14)
    ax[1].set_ylabel('Squared Residuals',fontsize=14)
    plt.tight_layout()
    plt.show()
