import pandas as pd
import numpy as np

def sanity_check(new_oos,state_space):
    print(f'State Space = {state_space}')
    model_list=['rfy_pred','svry_pred','xgb_abs_pred','xgb_q50_pred','xgb_huber_pred']
    model_preds=new_oos[model_list+['X_id']].copy()

    model_preds['model_risk']=model_preds[model_list].std(axis=1) ## for each fold-farm the disagreement across models
    farm_model_risk=model_preds[['model_risk','X_id']].groupby('X_id').mean() ## we take the over-fold-mean model-dissonance for each farm

    stacked = model_preds.set_index('X_id')[model_list].stack()
    farm_total_risk = stacked.groupby('X_id').std().to_frame('total_risk')

    farm_fold_risk = model_preds[['X_id']+model_list].groupby('X_id').std().mean(axis=1).to_frame('fold_risk') #for within model std for each farm and then the mean is the average over the models
    farm_risk_decomposition=pd.concat([farm_total_risk,farm_model_risk,farm_fold_risk],axis=1)
    farm_risk_decomposition['stability_ratio']=farm_risk_decomposition['model_risk']/farm_risk_decomposition['fold_risk']
    print('The model-to-fold stability and sanity check!')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(np.round(farm_risk_decomposition['stability_ratio'].describe(),4))
    stable_frac=((farm_risk_decomposition['stability_ratio']>1).mean())*100
    
    print(f'Percentage of farms where signals dominate noise :{stable_frac:.2f} \%')
    return farm_risk_decomposition
