from sklearn.metrics import precision_score, f1_score, roc_auc_score, accuracy_score, recall_score, precision_recall_curve, auc, average_precision_score
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

def event_metric_ARDS12h(event,inference_output,mode, model_name):
       
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'

    usecol = [icu_stay,'Time_since_ICU_admission', 'Annotation_ARDS', 'ARDS_next_12h', 'prediction_label']

    event_set = event[usecol[:-1]].copy()
    event_set['prediction_label'] = 'label'
    output_set = inference_output[usecol]
    output_set['prediction_label'] = output_set['prediction_label']

    event_set = event_set[event_set[icu_stay].isin(output_set[icu_stay].unique())]
    
    event_set_reset = event_set.reset_index(drop=True)
    output_set_reset = output_set.reset_index(drop=True)

    total_set = pd.concat([event_set_reset, output_set_reset], axis = 0, ignore_index=True).reset_index(drop=True).sort_values(by='Time_since_ICU_admission')

    total_event = event_set.Annotation_ARDS.value_counts()['ARDS']
    captured_event = 0

    captured_trajectory = []
    non_captured_trajectory = []

    True_positive = 0
    True__False_positive = 0

    for stayid in total_set[icu_stay].unique():
        
        interest = total_set[total_set[icu_stay]==stayid]
        
        if any(interest['ARDS_next_12h']=='event'):
            
            total_event += 1
            event_time = interest['Time_since_ICU_admission'].iloc[-1]
            time_window = event_time - 8
            capture_before_12h = interest[(interest['Time_since_ICU_admission'] >= time_window) & (interest['Time_since_ICU_admission'] < event_time)]
            
            if capture_before_12h['prediction_label'].sum() >= 1:
                captured_event += 1
                captured_trajectory.append(stayid)
            
            else:
                non_captured_trajectory.append(stayid)
                
    try:
        event_recall = np.round((captured_event / (len(captured_trajectory)+len(non_captured_trajectory)+0.00001)), 4)
    except:
        # print('ZeroDivisionError')
        event_recall = 0
    
    accuracy = accuracy_score(inference_output.ARDS_next_12h, inference_output.prediction_label)
    precision = precision_score(inference_output.ARDS_next_12h, inference_output.prediction_label)
    
    evaluation_score = {'acc':[accuracy],'recall':[event_recall], 'raw_precision':[precision]}
    
    
    return pd.DataFrame(evaluation_score, index=[model_name])

def ARDS12h_Event_AUPRC(event, inference_output, mode, model_name):
    
    if mode == 'mimic':
        icu_stay = 'stay_id'
    else:
        icu_stay = 'patientunitstayid'
        
    evaluation_interest = inference_output[inference_output[icu_stay].isin(event[icu_stay].unique())]
    
    y_true = evaluation_interest.prediction_label # 실제 레이블
    y_scores =evaluation_interest.Positive_Confscore # 예측 확률

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    new_recalls = []
    
    for threshold in tqdm(thresholds):
        evaluation_interest['prediction_label'] = evaluation_interest['Positive_Confscore'].apply(lambda x: 1 if x > threshold else 0).astype(int)
        result = event_metric_ARDS12h(event, evaluation_interest, mode, model_name)
        recall = result['recall'].values[0]
        new_recalls.append(recall)
    auprc = auc(np.array(new_recalls), precisions[:-1])
    return auprc


neg = pd.read_csv('./result/final_DA_Strict/negative_eicu_result.csv')
# pos = pd.read_csv('./result/final_DA_Strict/positive_eicu_result.csv')

result = neg
# result = pd.concat([neg, pos], axis = 0).sort_values(by=['patientunitstayid', 'Time_since_ICU_admission'])
result.drop(['uniquepid'], axis = 1, inplace = True)
result = result.reset_index(drop=True)

true, pred = result['ARDS_next_12h'] , result['pred']
result = result.rename(columns = {'pred':'prediction_label'})

# print(f'--- Evaluation Recall : {recall_score(true,pred)} | Precision : {precision_score(true,pred)} | Accuracy : {accuracy_score(true,pred)} | \n F1 Score : {f1_score(true,pred)} |AUC-ROC : {roc_auc_score(true,pred)}--- | AUPRC : {average_precision_score(true,pred)}' )

print(f'--- Evaluation Recall : {recall_score(true,pred)} | Precision : {precision_score(true,pred)} | Accuracy : {accuracy_score(true,pred)} | \n F1 Score : {f1_score(true,pred)} |--- | AUPRC : {average_precision_score(true,pred)}' )
