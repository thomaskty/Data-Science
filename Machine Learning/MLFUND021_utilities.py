import warnings
warnings.filterwarnings('ignore')
import pandas as pd
pd.set_option('display.float_format',lambda x : '%.3f' % x)
# pd.set_option('display.max_columns',10000)
# pd.set_option('display.max_colwidth',10000)
import numpy as np
# np.set_printoptions(suppress=True,precision=5,floatmode='fixed')
import sklearn
import sklearn as sk
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt 
from functools import * 
from scipy import stats
import dateutil
import datetime
import itertools
import pytz
from scipy import stats
from prettytable import PrettyTable


def get_current_time_ist():
    ist = pytz.timezone('Asia/Kolkata')
    now_utc = datetime.datetime.now(pytz.utc)
    now_ist = now_utc.astimezone(ist)
    formatted_time = now_ist.strftime('%Y-%m-%d %I:%M:%S %p')
    replacement_dict = {':': '', '-': '', ' ': ''}
    output = formatted_time
    for old, new in replacement_dict.items():
        output = output.replace(old, new)
    return output

def last_day(date):
    next_month = date.replace(day=28) + datetime.timedelta(days=4)
    return next_month - datetime.timedelta(days=next_month.day)

def get_yyyymm(dt):
    end_of_month = last_day(dt)
    year = str(end_of_month.year)
    month = f"{end_of_month.month:02d}" 
    mmyyyy = f"{month}{year}"
    yyyymm = f"{year}{month}"
    return mmyyyy, yyyymm

def get_previous_months(date_str, n):
    date = datetime.strptime(date_str, '%Y-%m-%d')  
    end_dates = []
    for _ in range(n):
        date = date - dateutil.relativedelta.relativedelta(months=1)  
        end_of_month = last_day(date)  
        end_dates.append(end_of_month.strftime('%Y-%m-%d')) 
    return tuple(end_dates)


def lag_n(input_date, n=-1):
    lagged_date = input_date + dateutil.relativedelta.relativedelta(months=n) 
    return last_day(lagged_date) 


def show_dateinfo():
    date_current = datetime.date.today()
    date_lag = lag_n(date_current, n=-1)
    month_name_current = date_current.strftime('%B')
    month_name_lag = date_lag.strftime('%B')
    year_current = date_current.year
    year_lag = date_lag.year
    mmyyyy_current, yyyymm_current = get_yyyymm(date_current)
    mmyyyy_lag, yyyymm_lag = get_yyyymm(date_lag)

    dd = pd.DataFrame(columns = ['Value'])

    dd.loc["date current"] = date_current
    dd.loc["date lag (-1)"] = date_lag
    dd.loc["month name current"] = month_name_current
    dd.loc["month name lag"] = month_name_lag
    dd.loc["year current"] = year_current
    dd.loc["year lag"] = year_lag
    dd.loc["mmyyyy_current"] = mmyyyy_current
    dd.loc["yyyymm_current"] = yyyymm_current
    dd.loc["mmyyyy_lag"] = mmyyyy_lag
    dd.loc["yyyymm_lag"] = yyyymm_lag
    dd.style.set_properties(**{'text-align': 'left'})
    table(dd)

def table(input_dataframe):
    df_or_series = input_dataframe.copy()
    if isinstance(df_or_series, pd.Series):
        index_name = df_or_series.index.name or 'index'
        series_name = df_or_series.name or 'value'
        df_or_series = df_or_series.reset_index()
        df_or_series.columns = [index_name, series_name]
    if isinstance(df_or_series.index, pd.MultiIndex):
        df_or_series.reset_index(inplace=True)
    else:
        df_or_series.reset_index(inplace=True)
    if isinstance(df_or_series.columns, pd.MultiIndex):
        df_or_series.columns = [
            '_'.join(map(str, col)).strip('_') for col in df_or_series.columns
        ]
    table = PrettyTable()
    table.field_names = df_or_series.columns.tolist()
    for row in df_or_series.itertuples(index=False):
        table.add_row(row)
    print(table)
    

def scale_data(data,method='standard',feature_range=(0,1)):
   if len(data.shape) ==1:
       data = data.reshape(-1,1)
   if method.lower()=='standard':
       scaler = preprocessing.StandardScaler()
   elif method.lower()=='minmax':
       scaler = preprocessing.MinMaxScaler(feature_range=feature_range)
   else:
       raise ValueError('Method must be either "standard" or "minmax"')
   scaled_data = scaler.fit_transform(data)
   return scaled_data,scaler

def inverse_scale_data(scaled_data,scaler):
   if len(scaled_data.shape)==1:
       scaled_data = scaled_data.reshape(-1,1)
   original_data = scaler.inverse_transform(scaled_data)
   return original_data

def get_combinations(param_dict):
   keys = param_dict.keys()
   values = param_dict.values()
   param_combinations = itertools.product(*values)
   param_dicts = [dict(zip(keys,combination)) for combination in param_combinations]
   return param_dicts

def optimal_nobins(data):
    n = len(data)
    min_max = data.max()-data.min()
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    h = 2*(iqr*(n**(-1/3)))
    return {'iqr':int(iqr/h),'min_max':int(min_max/h)}

def analyse_feature(df,feat1,feat2,relation,bins=False):
    if relation.lower() in ['cat2cat','con2cat']:
        print('feature analysis output is a dictionary with keys counts,total_ratio,bin_ratio')
    temp = df[[feat1,feat2]].copy()
    if relation.lower()=='cat2cat':
        output_dict = dict() 
        feat2_labels = temp[feat2].value_counts().index
        for i in feat2_labels:
            temp[i] = np.where(temp[feat2]==i,1,0)
        output1 = temp.groupby(feat1).agg({i:'sum' for i in feat2_labels})
        output1.loc['total'] = output1.sum()
        
        output2 = (output1/temp.shape[0])*100
        output3 = output1.div(output1.sum(axis=1), axis=0)
        output3 = (output3*100)
        
        output1['total'] = output1.sum(axis=1)
        output_dict['counts'] = output1 
        
        output2['total'] = output2.sum(axis=1)
        for j in output2.columns:
            output2[j] = output2[[j]].applymap(lambda x: "{0:.2f}%".format(x))
        output_dict['total_ratio'] = output2
        
        output3['total'] = output3.sum(axis=1)
        for j in output3.columns:
            output3[j] = output3[[j]].applymap(lambda x: "{0:.2f}%".format(x))
        output_dict['bin_ratio'] = output3 
        print('Counts') 
        table(output_dict['counts'])
        print('Total Ratio') 
        table(output_dict['total_ratio'])
        print('Binwise Ratio') 
        table(output_dict['bin_ratio'])
        
    elif relation.lower()=='con2con':
        # creating bins for feature a if not given
        if type(bins)==bool:
            nob = optimal_nobins(temp[feat1])['iqr']
            temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = nob,duplicates = 'drop')
        elif type(bins)==int:
            temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = bins,duplicates = 'drop')
        elif type(bins)==list:
            bins = pd.IntervalIndex.from_tuples(bins)
            temp[feat1+'_bins'] = pd.cut(temp[feat1],bins=bins,duplicates = 'drop')
        output1 = temp.groupby(feat1+'_bins').describe()[feat2]
        output1['25%'] = np.round(output1['25%'],3)
        output1['50%'] = np.round(output1['50%'],3)
        output1['75%'] = np.round(output1['75%'],3)
        output1['mean'] = np.round(output1['mean'],3)
        output1['std'] = np.round(output1['std'],3)
        output1['ratio'] = output1['count']/output1['count'].sum()
        output1['ratio'] = output1[['ratio']].applymap(lambda x: "{0:.2f}%".format(x*100))
        
        table(output1) 
    
    elif relation.lower()=='cat2con':
        output1 = temp.groupby(feat1).describe()[feat2]
        output1['25%'] = np.round(output1['25%'],3)
        output1['50%'] = np.round(output1['50%'],3)
        output1['75%'] = np.round(output1['75%'],3)
        output1['mean'] = np.round(output1['mean'],3)
        output1['std'] = np.round(output1['std'],3)
        
        output1['ratio'] = output1['count']/output1['count'].sum()
        output1['ratio'] = output1[['ratio']].applymap(lambda x: "{0:.2f}%".format(x*100))
        table(output1) 
        
    elif relation.lower()=='con2cat':
        output_dict = dict()
        if type(bins)==bool:
            nob = optimal_nobins(temp[feat1])['iqr']
            temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = nob,duplicates = 'drop')
        elif type(bins)==int:
            temp[feat1+'_bins'] = pd.qcut(temp[feat1],q = bins,duplicates = 'drop')
        elif type(bins)==list:
            bins = pd.IntervalIndex.from_tuples(bins)
            temp[feat1+'_bins'] = pd.cut(temp[feat1],bins=bins,duplicates = 'drop')
        
        feat2_labels = temp[feat2].value_counts().index
        for i in feat2_labels:
            temp[i] = np.where(temp[feat2]==i,1,0)
        output1 = temp.groupby(feat1+'_bins').agg({i:'sum' for i in feat2_labels})
        output1.loc['total'] = output1.sum()
        
        output2 = (output1/temp.shape[0])*100
        output3 = output1.div(output1.sum(axis=1), axis=0)
        output3 = (output3*100)
        
        output1['total'] = output1.sum(axis=1)
        output_dict['counts'] = output1
        
        output2['total'] = output2.sum(axis=1)
        for j in output2.columns:
            output2[j] = output2[[j]].applymap(lambda x: "{0:.2f}%".format(x))
        output_dict['total_ratio'] = output2
        
        output3['total'] = output3.sum(axis=1)
        for j in output3.columns:
            output3[j] = output3[[j]].applymap(lambda x: "{0:.2f}%".format(x))
        output_dict['bin_ratio'] = output3
        print('Counts') 
        table(output_dict['counts'])
        print('Total Ratio') 
        table(output_dict['total_ratio'])
        print('Binwise Ratio') 
        table(output_dict['bin_ratio'])

def thresh_analysis(model,x,y):
    cols = ['thresh','precision','recall','f1','accuracy','tp','fp','tn','fn']
    output = pd.DataFrame(columns=cols)
    thresh_list = [i/100 for i in list(range(30,80,1))]
    probs = model.predict_proba(x)[:,1]
    for i in range(len(thresh_list)):
        y_pred = np.where(probs>=thresh_list[i],1,0)
        cf = confusion_matrix(y,y_pred)
        tn,fp,fn,tp = cf[0,0],cf[0,1],cf[1,0],cf[1,1]
        p = tp/(tp+fp)
        r = tp/(tp+fn)
        f1 = (2*p*r)/(p+r)
        accuracy = (tp+tn)/(tn+fp+fn+tp)
        row = [thresh_list[i],p,r,f1,accuracy,tp,fp,tn,fn]
        output.loc[i] = row
    return output

def deciles(x,y,model):
    output = pd.DataFrame({'target':y,'proba':model.predict_proba(x)[:,1]})
    output['proba'] = np.round(output['proba'],3)
    output['decile'],probability_cuts = pd.qcut(
        output['proba'],q = 10,
        duplicates='drop',retbins=True
    )
    return output 

def aggregate(data):
    temp = data.copy()
    temp = temp.groupby('decile').agg({'target':['count',sum],'proba':['min','max']})
    temp.sort_index(ascending = False,inplace = True)
    temp['cust%'] = np.round(temp[('target','count')]/temp[('target','count')].sum(),3)
    temp['cum_cust%'] = np.round(temp['cust%'].cumsum(),3)
    temp['capture'] = np.round(temp[('target','sum')]/temp['target']['count'].sum(),3)
    temp['gain'] = np.round(temp[('target','count')]/temp[('target','sum')].sum(),3)
    temp['cum_gain'] = np.round(temp['gain'].cumsum(),3)
    temp['lift'] = np.round(temp['cum_gain']/temp['cum_cust%'].cumsum(),3)
    return temp.sort_index(ascending = False)

def prediction_frame(x,y,model,threshold):
    pred_df = pd.concat([x,y],axis=1)
    pred_df['proba'] = model.predict_proba(x)[:,1]
    pred_df['prediction'] = np.where(pred_df['proba']>threshold,1,0)
    pred_df['tp'] = np.where(((pred_df[y.name]==1)&(pred_df['prediction']==1)),1,0)
    pred_df['tn'] = np.where(((pred_df[y.name]==0)&(pred_df['prediction']==0)),1,0)
    pred_df['fp'] = np.where(((pred_df[y.name]==0)&(pred_df['prediction']==1)),1,0)
    pred_df['fn'] = np.where(((pred_df[y.name]==1)&(pred_df['prediction']==0)),1,0)
    pred_df['decile'] = pd.qcut(pred_df['proba'],q=10,duplicates='drop')
    return pred_df 

def population_stability_index(population,sample,model,name):
    def get_pred_fr(x,model):
        pred_df = x.copy()
        pred_df['proba'] = model.predict_proba(x)[:,1]
        return pred_df 
    pred_df_population = get_pred_fr(population,model)
    pred_df_sample = get_pred_fr(sample,model)
    pred_df_population['psi_decile'],proba_cuts = pd.qcut(
        pred_df_population['proba'],
        q = 10,
        duplicates='drop',
        retbins=True
    )
    proba_cuts[0] = 0 
    proba_cuts[-1] = 1
    pred_df_population['psi_decile'] = pd.cut(pred_df_population['proba'],bins = proba_cuts)
    pred_df_sample['psi_decile'] = pd.cut(pred_df_sample['proba'],bins = proba_cuts)
    score_b = pred_df_sample.groupby('psi_decile').agg({
        'proba':'count'
    }).sort_index(ascending = False)
    score_a = pred_df_population.groupby('psi_decile').agg({
        'proba':'count'
    }).sort_index(ascending = False)
    score_a.columns = ['population_A']
    score_b.columns = [name]
    psi_df = pd.concat([score_a,score_b],axis=1).sort_index()
    psi_df['A%'] = np.round((psi_df['population_A']/psi_df['population_A'].sum())*100)
    psi_df['B%'] = np.round((psi_df[name]/psi_df[name].sum())*100)
    psi_df['A% - B%'] = psi_df['A%']- psi_df['B%']
    psi_df['log(A%/B%)'] = np.log(psi_df['A%']/psi_df['B%'])
    psi_df['A%'] = psi_df[['A%']].applymap(lambda x: "{0:.0f}%".format(x))
    psi_df['B%'] = psi_df[['B%']].applymap(lambda x: "{0:.0f}%".format(x))
    psi_df['psi'] = (psi_df['A% - B%']/100)*psi_df['log(A%/B%)']
    psi_df['A% - B%'] = psi_df[['A% - B%']].applymap(lambda x: "{0:.0f}%".format(x))
    psi_df.sort_index(ascending = False,inplace = True)
    return psi_df

def auc_gini(x,y,model):
    preds = model.predict_proba(x)[:,1]
    fpr,tpr,thresholds = roc_curve(y,preds)
    auc_ = auc(fpr,tpr)
    gini = (2*auc_)-1
    return {'auc':auc_,'gini':gini}

def error_analysis_feature_bins(x,y,model,t,f,bins):
    pred_df = prediction_frame(x,y,model,t)
    if type(bins)==int:
        pred_df[f]= pd.qcut(pred_df[f],q = bins,duplicates = 'drop')
    elif type(bins)==str:
        pred_df[f] = pd.qcut(pred_df[f],q = optimal_nobins(pred_df[f])['iqr'],duplicates='drop')
    elif type(bins)==list:
        bins = pd.IntervalIndex.from_tuples(bins)
        pred_df[f] = pd.cut(pred_df[f],bins = bins)
    mc = pred_df.groupby(f).agg({i:sum for i in ['tn','tp','fn','fp']}).reset_index()
    mc['+ preds'] = mc[['tp','fp']].sum(axis=1)
    mc['- preds'] = mc[['tn','fn']].sum(axis=1)
    mc['+ actual'] = mc[['tp','fn']].sum(axis=1)
    mc['- actual'] = mc[['tn','fp']].sum(axis=1)
    mc['true preds'] = mc[['tn','tp']].sum(axis=1)
    mc['false preds'] = mc[['fp','fn']].sum(axis=1)
    mc['precision'] = mc['tp']/mc['+ preds']
    mc['recall'] = mc['tp'] /mc['+ actual']
    mc['f1'] = (2*mc['precision']*mc['recall'])/(mc['precision']+mc['recall'])
    mc['accuracy'] = mc[['tp','tn']].sum(axis=1)/mc[['tp','tn','fn','fp']].sum(axis=1)
    return mc 

def get_metrices(x,y,model,thresh):
    pred_df = prediction_frame(x,y,model,thresh)
    dict_final = {}
    for i in ['tp','tn','fp','fn']:
        dict_final[i] = pred_df[i].sum()
    tp,tn = dict_final['tp'],dict_final['tn']
    fp,fn = dict_final['fp'],dict_final['fn']
    precision = np.round(tp/(tp+fp),2)
    recall = np.round(tp/(tp+fn),2)
    f1 = np.round((2*precision*recall)/(precision+recall),2)
    accuracy = np.round((tp+tn)/(tp+fn+fp+fn),2)
    return {'precision':precision,'recall':recall, 'f1':f1,'accuracy':accuracy}


def confusion_matrix_formatted(x,y,model,name,thresh):
    pred_df = prediction_frame(x,y,model,thresh)
    dict_final = {}
    for i in ['tp','tn','fp','fn']:
        dict_final[i] = pred_df[i].sum()
    d = pd.DataFrame(index = [0,1],columns= pd.MultiIndex.from_tuples([
        ('{}_{}'.format(name,thresh),'Predicted',0),
        ('{}_{}'.format(name,thresh),'Predicted',1)
    ]))
    d.index.name = 'Actual'
    d.loc[0] = [dict_final['tn'],dict_final['fp']]
    d.loc[1] = [dict_final['fn'],dict_final['tp']]
    d.loc['Total'] = [int(i) for i in d['{}_{}'.format(name,thresh)]['Predicted'].sum()]
    d[('{}_{}'.format(name,thresh),'Predicted','total')] = d.sum(axis=1)
    return d

def calculate_psi(expected,actual,buckettype='bins',buckets=10,axis=0):
    def psi(expected_array,actual_array,buckets):
        def scale_range(input_,min_,max_):
            input_ +=-(np.min(input_))
            input_ /=np.max(input_)/(max_-min_)
            input_ +=min
            return input_
        breakpoints = np.arange(0,buckets+1)/(buckets)*100
        if buckettype=='bins':
            breakpoints = scale_range(breakpoints,np.min(expected_array),np.max(expected_array))
        elif buckettype=='quantiles':
            breakpoints = np.stack([np.percentile(expected_array,b) for b in breakpoints])
        expected_percents = np.histogram(expected_array,breakpoints)[0]/len(expected_array)
        actual_percents = np.histogram(actual_array,breakpoints)[0]/len(actual_array)
        def sub_psi(e_perc,a_perc):
            if a_perc ==0:
                a_perc= 0.0001
            if e_perc ==0:
                e_perc=0.0001
            value = (e_perc - a_perc)*np.log(e_perc/a_perc)
            return (value)
        psi_value = np.sum(sub_psi(
            expected_percents[i],actual_percents[i]
        ) for i in range(0,len(expected_percents)))

        return (psi_value)
    if len(expected.shape)==1:
        psi_values = np.empty(len(expected.shape))
    else:
        psi_value = np.empty(expected.shape[axis])
    for i in range(0,len(psi_values)):
        if len(psi_values)==1:
            psi_values = psi(expected,actual,buckets)
        elif axis==0:
            psi_values[i] = psi(expected[:,i],actual[:,i],buckets)
        elif axis==1:
            psi_values[i] = psi(expected[i,:],actual[i,:],buckets)
    return (psi_values)


def calculate_psi_category(initial,new):
    df1 = pd.DataFrame(initial.value_counts(normalize = True).reset_index())
    df2 = pd.DataFrame(new.value_counts(normalize = True).reset_index())
    df = pd.merge(df1,df2,how = 'left',on='index')
    df.columns = ['index','initial_perc','new_perc']
    df.fillna(0.0001,inplace = True)
    df['PSI'] = (df['new_perc']-df['initial_perc'])*np.log(df['new_perc']/df['initial_perc'])
    return np.round(np.sum(df['PSI']),5)

def psi_alert_label(psi):
    return np.where(psi<0.1,'low',np.where(psi>=0.2,'high','medium'))

def calculate_covariate_drift(old_data,new_data):
    li_common_cols = old_data.columns.intersection(new_data.columns)
    old_data = old_data[li_common_cols]
    new_data = new_data[li_common_cols]
    num_cols = old_data._get_numeric_data().columns
    obj_cols = list(set(li_common_cols)-set(num_cols))
    t = pd.DataFrame(columns = ['PSI Value','Drift Level'])
    
    for var in num_cols:
        psi = calculate_psi(old_data[var],new_data[var],buckettype='quantiles',buckets = 10,axis=1)
        t.loc[var] = [psi,psi_alert_label(psi)]
    for  var in obj_cols:
        psi = calculate_psi_category(old_data[var],new_data[var])
        t.loc[var] = [psi,psi_alert_label(psi)]
    return t


def detect_model_drift(model,data_old,y_old,data_new,y_new,tolerance,thresh):
    model_score_old = get_metrices(data_old,y_old,model,thresh)
    f1_old = model_score_old['f1']
    precision_old = model_score_old['precision']
    recall_old = model_score_old['recall'] 

    new_scores = get_metrices(data_new,y_new,model,thresh)
    f1_new = new_scores['f1']
    precision_new = new_scores['precision']
    recall_new = new_scores['recall']

    precision_flag = False
    recall_flag = False
    f1_flag = False
    if f1_new<(f1_old-f1_old*tolerance):
        f1_flag=True
    if precision_new <(precision_old-precision_old*tolerance):
        precision_flag = True
    if recall_new <(recall_old-recall_old*tolerance):
        recall_flag = True
    t = pd.DataFrame(columns = ['Drift Observed'])
    t.loc['f1'] = [f1_flag]
    t.loc['recall'] = [recall_flag]
    t.loc['precision'] = [precision_flag]
    return t 

def calculate_residual_drift(model,x_new,y_new,x_old,y_old):
    """
    for regression models
    """
    old_predictions = model.predict(x_old)
    new_predictions = model.predict(x_new)
    residuals_old = y_old - old_predictions 
    residuals_new = y_new - new_predictions

    psi = calculate_psi(residuals_old,residuals_new,buckettype='quantiles',buckets=10,axis=1)
    output = pd.DataFrame(columns = ['psi_value','drift_level'])
    output.loc['observed_residual_shift'] = [psi,psi_alert_label(psi)]
    return output


def entropy(values, probability=False):   
    if probability:
        return -np.sum(values * np.log2(values + np.finfo(float).eps))
    else:
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)
        return -np.sum(probs * np.log2(probs + np.finfo(float).eps))


def compute_probabilities(labels):
    # Get unique labels and their respective counts
    unique_labels, counts = np.unique(labels, return_counts=True)
    # Calculate probabilities
    probabilities = counts / np.sum(counts)
    # Map each unique label to its probability
    out = dict(zip(unique_labels, probabilities))
    return out

def cross_entropy(p, q, probabilities=False, verbose=True):
    if not probabilities:
        labels_p = list(np.unique(p))
        labels_q = list(np.unique(q))
        # Determine the union of labels from both p and q
        final_labels = labels_q if len(labels_q) > len(labels_p) else labels_p
        # Compute probabilities for p and q
        p = compute_probabilities(p)
        q = compute_probabilities(q)
        
        # Handle missing labels
        for label in final_labels:
            if label not in p:
                p[label] = 0
            if label not in q:
                q[label] = 0
        
        if verbose:
            print(f"Computed probabilities for p: {p}")
            print(f"Computed probabilities for q: {q}")
            print(f"Final labels: {final_labels}")
        
        # Compute cross-entropy
        cross_entropy_measure = []
        for label in final_labels:
            if q[label] != 0:
                cross_entropy_measure.append(p[label] * np.log2(1 / q[label]))
            else:
                cross_entropy_measure.append(0)
    else:
        # Directly compute cross-entropy if probabilities are given
        p = np.array(p)
        q = np.array(q)
        
        cross_entropy_measure = p * np.log2(1 / q)
        cross_entropy_measure[np.isinf(cross_entropy_measure)] = 0  # Handle log(0)
    
    if verbose:
        print(f"Cross-entropy measure: {cross_entropy_measure}")
    
    return np.sum(cross_entropy_measure)


def joint_entropy(x, y):
    # Get unique values and counts for x and y
    x_values, x_counts = np.unique(x, return_counts=True)
    y_values, y_counts = np.unique(y, return_counts=True)

    # Create joint frequency matrix
    joint_counts = np.zeros((len(x_values), len(y_values)), dtype=int)
    for i in range(len(x)):
        x_idx = np.where(x_values == x[i])[0][0]
        y_idx = np.where(y_values == y[i])[0][0]
        joint_counts[x_idx, y_idx] += 1
    
    # Calculate joint probabilities
    joint_probs = joint_counts / len(x)

    # Compute joint entropy, handling zero probabilities
    return -np.sum(joint_probs * np.log2(joint_probs + (joint_probs == 0)))

def mutual_information_value(x, y, verbose=False):
    x.name = 'feature'
    y.name = 'target'
    if type(y) != pd.core.series.Series:
        y = pd.Series(y, name='target')
        
    if type(x) != pd.core.series.Series:
        x = pd.Series(x, name='feature')
    
    # Create a contingency table (cross-tabulation)
    output = pd.crosstab(x, y, margins=True, margins_name='total')
    if verbose:
        print(output)
    output /= len(x)
    if verbose:
        print(output)
    def compute_mi(join, marginals):
        return join * np.log2(join / reduce(lambda x, y: x * y, marginals))
    total_mi = []
    rows = len(output.index) - 1
    cols = len(output.columns) - 1
    for r in range(rows):
        for c in range(cols):
            value = output.iloc[r, c]
            marg_a, marg_b = output.iloc[r, cols], output.iloc[rows, c]
            if verbose:
                print(value, [marg_a, marg_b], [r, c])
            if value != 0:
                mi_value = compute_mi(value, [marg_a, marg_b])
            else:
                mi_value = 0
            # Handle infinity values (e.g., log(0)) by setting them to zero
            if mi_value == np.inf:
                mi_value = 0
            total_mi.append(mi_value)
    if verbose:
        print(total_mi)
    return np.sum(total_mi)


def mutual_information_frame(df, target_name):
    cdata = df.copy()
    temp_df = cdata.drop(target_name, axis=1)
    temp_y = cdata[target_name]
    temp_new = pd.DataFrame()
    try:
        object_cols = cdata.describe(include='O').T.index.tolist()
    except Exception:
        object_cols = []
    cols_categorized = [i for i in cdata.columns if i not in object_cols + [target_name]]
    for j in cols_categorized:
        try:
            nob = optimal_nobins(temp_df[j])['iqr']
            temp_new[j + '_bin'] = pd.qcut(temp_df[j], q=nob, duplicates='drop')
        except Exception as e:
            print(f"Error binning column '{j}': {e}")
    
    X, y = temp_new, temp_y
    out_dict = {'features': X.columns, 'mi_score': []}
    for i in X.columns:
        try:
            mi_score = mutual_information_value(X[i], y)
            out_dict['mi_score'].append(mi_score)
        except Exception as e:
            print(f"Error calculating MI for feature '{i}': {e}")
            out_dict['mi_score'].append(None)
    return pd.DataFrame(out_dict)

def kl_divergence(p, q, probabilities=False, verbose=True):
    if not probabilities:
        labels_p = list(np.unique(p))
        labels_q = list(np.unique(q))
        final_labels = list(set(labels_p) | set(labels_q))
        p = compute_probabilities(p)
        q = compute_probabilities(q)

        for label in final_labels:
            p.setdefault(label, 0)
            q.setdefault(label, 0)
        if verbose:
            print(f"Probabilities (p): {p}")
            print(f"Probabilities (q): {q}")
            print(f"Labels: {final_labels}")

        kl_divergence_measure = []
        for label in final_labels:
            if q[label] != 0:
                kl_divergence_measure.append(p[label] * np.log2(p[label] / q[label]))
            else:
                kl_divergence_measure.append(0)
    else:
        p = np.array(p)
        q = np.array(q)
        kl_divergence_measure = p * np.log2(p / q)
        kl_divergence_measure[np.isinf(kl_divergence_measure)] = 0
    if verbose:
        print(f"KL Divergence Components: {kl_divergence_measure}")
    return np.sum(kl_divergence_measure)


def calculate_kendall_tau_trend(data):
    x = np.arange(len(data))
    tau,_ = stats.kendalltau(x,data)
    return tau

def average_monthly_growth(data):
    mom_growth = np.diff(data)/data[1:]*100
    return np.mean(mom_growth)

def cumulative_growth(data):
    return ((data[0]-data[-1])/data[-1])*100

def calculate_coefficient_variation(data):
    mean = np.mean(data)
    std = np.std(data)
    cv = (std/mean)*100 if mean !=0 else 0
    return cv

def recentavg_vs_pastavg_ratio(data):
    # index 0 should be the oldest data
    # [m6,m5,m4,m3,m2,m1]
    # past data = [m6,m5,,m4]
    # latest data = [m3,m2,m1]
    middle_point = len(data)//2
    if np.mean(data[middle_point:])!=0:
        return np.mean(data[:middle_point])/np.mean(data[middle_point:])
    else:
        return 0

def recentsum_vs_pastsum_ratio(data):
    # index 0 should be the oldest data
    # [m6,m5,m4,m3,m2,m1]
    # past data = [m6,m5,,m4]
    # latest data = [m3,m2,m1]
    middle_point = len(data)//2
    if np.sum(data[middle_point:])!=0:
        return np.sum(data[:middle_point])/np.sum(data[middle_point:])
    else:
        return 0

def peak_count(data):
    mean_value = np.mean(data)
    peak_count_ = np.sum(data>mean_value)
    return peak_count_

def trough_count(data):
    mean_value = np.mean(data)
    trough_count_ = np.sum(data<mean_value)
    return trough_count_

def decay_weighted_average(data,decay_rate=0.1):
    n = len(data)
    weights = np.exp(-decay_rate*np.arange(n))
    weighted_avg = np.sum(weights*data)/np.sum(weights)
    return weighted_avg

def optimal_nobins(data):
    n = len(data)
    min_max = data.max()-data.min()
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    h = 2*(iqr*(n**(-1/3)))
    return {'iqr':int(iqr/h),'min_max':int(min_max/h)}

def remove_correlated_features_variance_method(df, threshold=0.6):
   features_to_keep = list(df.columns)
   iteration = 0
   log_data = []
   correlated_features_dict = {f: set() for f in features_to_keep}
   all_features_dropped = set()

   while True:
       iteration += 1
       corr_matrix = df[features_to_keep].corr().abs()
       upper_triangle =corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
       correlated_pairs = [
           (col, upper_triangle.index[idx])
           for idx, col in enumerate(upper_triangle.columns)
           for col in upper_triangle.index[upper_triangle[col] > threshold]
       ]
       if not correlated_pairs:
           break
       drop_candidates = set()
       for feature_1, feature_2 in correlated_pairs:
           variance_1 = df[feature_1].var()
           variance_2 = df[feature_2].var()
           correlated_features_dict[feature_1].add(feature_2)
           correlated_features_dict[feature_2].add(feature_1)
           to_drop = feature_1 if variance_1 < variance_2 else feature_2
           drop_candidates.add(to_drop)
           log_data.append({
               "Feature1": feature_1,
               "Feature2": feature_2,
               "Feature1_Variance": np.round(variance_1,3),
               "Feature2_Variance": np.round(variance_2,3),
               "Correlation": np.round(corr_matrix.loc[feature_1, feature_2],3),
               "Dropping": to_drop,
               "Iteration_Step": iteration,
               "Feature1_Correlated_With_Count":len(correlated_features_dict[feature_1]),
               "Feature2_Correlated_With_Count":len(correlated_features_dict[feature_2]),
               "Feature1_Correlated_With_List":list(correlated_features_dict[feature_1]),
               "Feature2_Correlated_With_List":list(correlated_features_dict[feature_2]),
           })
       features_to_keep = [f for f in features_to_keep if f not in drop_candidates]
       all_features_dropped.update(drop_candidates)
   log_df = pd.DataFrame(log_data)
   return list(all_features_dropped),log_df

def remove_correlated_features_target(df, target, threshold=0.6):
    features = list(df.columns)
    features.remove(target)  # Remove target from the list of features
    features_to_keep = features.copy()
    log_data = []
    iteration = 0
    correlated_features_dict = {f: set() for f in features_to_keep}
    all_features_dropped = set()

    while True:
        iteration += 1
        corr_matrix = df[features_to_keep].corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_pairs = [
            (col, upper_triangle.index[idx])
            for idx, col in enumerate(upper_triangle.columns)
            for col in upper_triangle.index[upper_triangle[col] > threshold]
        ]

        if not correlated_pairs:
            break
        drop_candidates = set()
        for feature_1, feature_2 in correlated_pairs:
            target_corr_1 = df[feature_1].corr(df[target])
            target_corr_2 = df[feature_2].corr(df[target])
            correlated_features_dict[feature_1].add(feature_2)
            correlated_features_dict[feature_2].add(feature_1)
            to_drop = feature_1 if abs(target_corr_1) < abs(target_corr_2) else feature_2
            drop_candidates.add(to_drop)
            log_data.append({
                "Feature1": feature_1,
                "Feature2": feature_2,
                "Feature1_Target": np.round(target_corr_1,3),
                "Feature2_Target": np.round(target_corr_2,3),
                "Correlation": np.round(corr_matrix.loc[feature_1, feature_2],3),
                "Dropping": to_drop,
                "Iteration_Step": iteration,
                "Feature1_Correlated_With_Count": len(correlated_features_dict[feature_1]),
                "Feature2_Correlated_With_Count": len(correlated_features_dict[feature_2]),
                "Feature1_Correlated_With_List": list(correlated_features_dict[feature_1]),
                "Feature2_Correlated_With_List": list(correlated_features_dict[feature_2]),
            })
        features_to_keep = [f for f in features_to_keep if f not in drop_candidates]
        all_features_dropped.update(drop_candidates)
    log_df = pd.DataFrame(log_data)
    return list(all_features_dropped), log_df

def woe_iv(data,feature,target,events,bins,continuous):
    temp = data[[feature,target]].copy()
    if continuous:
        if type(bins)==str:
            temp[feature+'_bins'] = pd.qcut(
                temp[feature],
                q = optimal_nobins(temp[feature])['iqr'],
                duplicates='drop'
            )
        elif type(bins)==list:
            bins = pd.IntervalIndex.from_tuples(bins)
            temp[feature+'_bins'] = pd.cut(temp[feature],bins)
        elif type(bins)==int:
            temp[feature+'_bins'] = pd.qcut(temp[feature],q = bins,duplicates='drop')
        feature = feature+'_bins'
    target_labels = temp[target].value_counts().index 
    temp[events] = np.where(temp[target]==events,1,0)
    temp = temp.groupby(feature).agg({events:['sum','count']})
    temp['%events'] = np.round(temp[events]['sum']/temp[events]['sum'].sum(),3)
    temp['#nonevents'] = temp[events]['count']-temp[events]['sum']
    temp['%nonevents'] = np.round(temp['#nonevents']/temp['#nonevents'].sum(),3)
    temp['woe'] = np.round(np.log((temp['%events']) / (temp['%nonevents'])),3)
    temp['IV'] = np.round(temp['woe']* (temp['%events'] - temp['%nonevents']),3)
    table(temp) 
    return temp

def chi_selection(data, target_name):
    cdata = data.copy()
    from scipy import stats
    temp_df = cdata.drop(target_name, axis=1)
    temp_y = cdata[target_name]
    temp_new = pd.DataFrame()
    try:
        object_cols = cdata.describe(include='O').T.index.tolist()
    except Exception as e:
        print(f"Error in identifying categorical columns: {e}")
        object_cols = []
    
    cols_categorized = [i for i in cdata.columns if i not in object_cols + [target_name]]
    
    for j in cols_categorized:
        try:
            mode_j = temp_df[j].mode().values[0]
            temp_df[j].fillna(mode_j, inplace=True)
            nob = optimal_nobins(temp_df[j])['iqr']
            temp_new[j + '_bin'] = pd.qcut(temp_df[j], q=nob, duplicates='drop')
        except Exception as e:
            print(f"Error processing feature {j}: {e}")
            cols_categorized.remove(j)
    try:
        temp_df.fillna(0, inplace=True)
    except Exception as e:
        print(f"Error while filling missing values: {e}")
    
    X, y = temp_new, temp_y
    p_values = []
    chi_square_stats = []
    for feature in X.columns:
        try:
            chi2_stat, p_val, _, _ = stats.chi2_contingency(pd.crosstab(X[feature], y))
            p_values.append(p_val)
            chi_square_stats.append(chi2_stat)
        except Exception as e:
            print(f"Error calculating Chi-Square for feature {feature}: {e}")
            p_values.append(None)
            chi_square_stats.append(None)
    out = pd.DataFrame({
        'feature': X.columns,
        'p_value': p_values,
        'chi_square_stats': chi_square_stats
    })
    out = out.sort_values(by='chi_square_stats', ascending=True).set_index('feature')
    return out



def get_data_sample(data):
    df = data.copy()
    np.random.seed(42)
    subset_fraction  = 0.05 
    subset_size = int(df.shape[0]*subset_fraction)
    subset_indices = np.random.choice(df.shape[0],subset_size,replace = False)
    data_subset = df[subset_indices]
    return data_subset,subset_indices
    