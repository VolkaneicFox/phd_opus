import matplotlib.pyplot as plt
import pandas as pd

def plot_coverage(preds,full_conf,up_conf,low_conf,mean_yield,max_band,min_band):

    print(f'Confidence Bands are between {min_band} and {max_band} Standard Deviations')
    coverage_text = f"Coverage = {full_conf}%\nLeft tail = {low_conf}%\nRight tail = {up_conf}%"
    fig,ax=plt.subplots(figsize=(12,5))
    preds=preds.sort_values(by='bands_i',ascending=True).reset_index(drop=True)
    ax.vlines(x=preds['bands_i'],ymin=preds[f'xgb_q{lband}_pred'],ymax=preds[f'xgb_q{uband}_pred'],
              color='skyblue',alpha=0.8,label='XGB Quantile Bands')
    ax.scatter(preds['bands_i'], preds['yeo_farm_crop_yield'], color='crimson', s=2, zorder=2, label='Observed Yield')
    ax.plot(preds['bands_i'],[mean_yield]*preds.shape[0],color='green',alpha=0.3)
    ax.text(
        0.02, 0.02,  # relative position inside the subplot
        coverage_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )
    ax.set_ylabel('Observed Farm Yields',fontsize=10)
    ax.set_xlabel('95-5th Percentile Bands normalized by Standard Deviation',fontsize=10)
    plt.suptitle('Farm Yield Prediction Bands',fontsize=13)
    plt.legend()
    plt.show()

random_states=[3504,2992,3529,5042,4053,2492,9193,6843,3923,5935]
def make_quantile_coverage(new_oos,ensemble):
    coverage_list = []
    uband, lband = '95', '05'
    for fold_no in range(1,16):
        cols = ['X_id', f'xgb_q{uband}_pred', f'xgb_q{lband}_pred', 'yeo_farm_crop_yield','xgb_q50_pred']
        preds = new_oos[new_oos['fold'] == fold_no][cols].copy()
        preds = preds.reset_index().rename(columns={'index':'X_id'})
        preds['in_upper'] = (preds['yeo_farm_crop_yield'] <= preds[f'xgb_q{uband}_pred']).astype(int)
        preds['in_lower'] = (preds['yeo_farm_crop_yield'] >= preds[f'xgb_q{lband}_pred']).astype(int)
        preds['full_conf_bool'] = (preds['in_upper'] & preds['in_lower']).astype(int)
        full_conf = np.round(preds['full_conf_bool'].mean() * 100, 2)
        up_conf = np.round(preds['in_upper'].mean() * 100, 2)
        low_conf = np.round(preds['in_lower'].mean() * 100, 2)


    coverage = pd.concat(coverage_list, axis=0).reset_index(drop=True)
    return coverage
