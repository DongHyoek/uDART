import pandas as pd
import numpy as np

def get_ARDS(target, event='ARDS', mode = 'mimic'):
    
    data = target.copy()
    
    split_data = []
    current_part = []
    event_occurred = False
    
    if mode == 'mimic':
        stay_id = 'stay_id'
    else:
        stay_id = 'patientunitstayid'
        
    search_stay_id = set(data[stay_id].unique())
    
    for stayid in search_stay_id:
        dataset = data[data[stay_id]==stayid]
        
        for index, row in dataset.iterrows():
            
            if event_occurred:
                event_occurred = False
                break
                        
            else:
                current_part.append(row)
                if row['Annotation']==event:
                    split_data.append(pd.DataFrame(current_part))
                    event_occurred = True
                    current_part = []
            
        if current_part:
            split_data.append(pd.DataFrame(current_part))
            current_part = []

    return pd.concat(split_data).reset_index(drop=True)

if __name__ == "__main__":
    print(f'MIMIC / eICU Cut off By Event!!')
    eicu = pd.read_csv('./dataset/eicu_ARDS.csv')
    mimic = pd.read_csv('./dataset/mimic_ARDS.csv')

    mimic = get_ARDS(mimic, event='ARDS', mode = 'mimic')
    eicu = get_ARDS(eicu, event='ARDS', mode = 'eicu')
    
    print('Save..')
    mimic.to_csv('./dataset/mimic_ARDS_cutoff.csv', index = False)
    eicu.to_csv('./dataset/eicu_ARDS_cutoff.csv', index = False)
    
    print('Done..')