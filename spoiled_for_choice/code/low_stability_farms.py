import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import ensemble_predictor_and_metrics

def make_low_stable_farms_dispersion(state_files_path):
    random_states=[3504,2992,3529,5042,4053,2492,9193,6843,3923,5935]
    models_list=['random-forest','svr','xgb_abs','xgb_q50','xgb_huber']
    ensemble_prediction=[]
    candidate_performances=[]
    fin_metrics=[]
    for state_space in random_states:
        print(state_space)
        infile=pd.read_csv(f'{state_files_path}preds_{state_space}.csv')
        infile=make_xid(infile.copy(),'rice')
        inr2=pd.read_csv(f'{state_files_path}r2_{state_space}.csv')
        inr2=inr2[models_list]
        kappa=2
        wts=np.exp(kappa*(inr2.sub(inr2.max(axis=1),axis=0)))
        wts_star=wts.div(wts.sum(axis=1),axis=0).mean(axis=0)
        print('The sum of normalized weights', round(wts_star.sum()))
        check_ovf=infile[['X_id','fold','rfy_pred', 'svry_pred', 'xgb_abs_pred', 'xgb_q50_pred', 'xgb_huber_pred']].copy().sort_values(by=['X_id','fold']).drop('fold',axis=1).reset_index(drop=True)
        check_ovf=check_ovf.groupby('X_id').std()
        check_ovf=pd.concat([check_ovf,
                             infile[['X_id','yeo_farm_crop_yield']].drop_duplicates().set_index('X_id')],axis=1)
        check_ovf=check_ovf.div(check_ovf.yeo_farm_crop_yield,axis=0)
        check_ovf=abs(check_ovf)
        check_ovf_qtls=pd.concat([pd.qcut(check_ovf['rfy_pred'],q=4,labels=[1,2,3,4]),
               pd.qcut(check_ovf['svry_pred'],q=4,labels=[1,2,3,4]),
               pd.qcut(check_ovf['xgb_abs_pred'],q=4,labels=[1,2,3,4]),
               pd.qcut(check_ovf['xgb_q50_pred'],q=4,labels=[1,2,3,4]),
               pd.qcut(check_ovf['xgb_huber_pred'],q=4,labels=[1,2,3,4])],axis=1)
        low_stable_farms_list=list(check_ovf_qtls[(check_ovf_qtls.min(axis=1)>=3)].index)
        low_stable_farms=infile[infile['X_id'].isin(low_stable_farms_list)].copy()
        low_stable_farms=low_stable_farms[['X_id','fold','yeo_farm_crop_yield','rfy_pred', 'svry_pred', 'xgb_abs_pred', 
                                       'xgb_q50_pred', 'xgb_huber_pred']].sort_values(by=['X_id','fold'])
        temp=make_ensemble_predictions(low_stable_farms,wts_star)
        ran=temp[0].rename(columns={'y_star':f'ran_ypred'})
        var=temp[1].rename(columns={'y_star':f'var_ypred'})
        mea=temp[2].rename(columns={'y_star':f'mea_ypred'})

        tempdf=pd.concat([ran[['y_obs',f'ran_ypred']],
                          var[[f'var_ypred']],
                          mea[[f'mea_ypred']]],
                        axis=1) ## ensemble predictions

        tempdf['state_space']=state_space
        ensemble_prediction.append(tempdf.reset_index())
        perform_dict=candidate_performance(low_stable_farms)
        perform_df=pd.DataFrame(perform_dict)
        perform_df.index=models_list ## candidate performance
        perform_df=perform_df.T
        perform_df['state_space']=state_space
        candidate_performances.append(perform_df.reset_index())
        y_obs=tempdf['y_obs'].values
        metrics_dict={'ran_r2':[r2_score(y_obs,tempdf[f'ran_ypred'])],
                      'ran_rmse':[mean_squared_error(y_obs,tempdf[f'ran_ypred'],squared=False)],
                      'ran_mae':[mean_absolute_error(y_obs,tempdf[f'ran_ypred'])],
                      'var_r2':[r2_score(y_obs,tempdf[f'var_ypred'])],
                      'var_rmse':[mean_squared_error(y_obs,tempdf[f'var_ypred'],squared=False)],
                      'var_mae':[mean_absolute_error(y_obs,tempdf[f'var_ypred'])],
                      'mea_r2':[r2_score(y_obs,tempdf[f'mea_ypred'])],
                      'mea_rmse':[mean_squared_error(y_obs,tempdf[f'mea_ypred'],squared=False)],
                      'mea_mae':[mean_absolute_error(y_obs,tempdf[f'mea_ypred'])]}
        metrics_dict['state_space']=[state_space]
        metrics_df=pd.DataFrame(metrics_dict)
        fin_metrics.append(metrics_df)
    return {'ensemble':ensemble_prediction,'candidate':candidate_performances,'metrics':fin_metrics}
def calculate_percentage_change(mean_metric,b):
    '''
    calculate the improvement of b over the mean metric
    '''
    output=((mean_metric-b)*100)/mean_metric
    return np.round(output,2)
