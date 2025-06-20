#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import gc

pd.set_option('mode.chained_assignment',  None)

mimic_path = "./dataset/mimic_ARDS.csv.gz"
eicu_path = "./dataset/eicu_ARDS.csv.gz"

def replace_inf(df,col,method='zero'):
    df[col].replace([np.inf, -np.inf], np.nan, inplace = True)

    if method=='zero':
        df[col].fillna(0,inplace = True)
    else:
        max_value = df[col].max()
        df[col].fillna(max_value,inplace = True)
    return df

def integration_source_target(mimic_path, eicu_path):

    mimic = pd.read_csv(mimic_path, compression='gzip', index_col = 0)
    eicu  = pd.read_csv(eicu_path, compression='gzip', index_col = 0)
    # 잘못 표기한 컬럼 명 변경 및 삭제

    mimic = mimic.rename(columns={'SaO2': 'O2 Sat (%)', 'SaO2_fillna':'O2 Sat (%)_fillna'})
    eicu = eicu.drop(['SaO2', 'SaO2_fillna'], axis = 1)
    # 필요 없는 컬럼 날리기

    eicu = eicu.drop(['Antibiotic(lab)', 'CT scan', 'PAOP', 'PAOP_fillna', 'Vasopressin (MCG/KG/MIN)_Rate', 'med_Nitroprusside',
            'Norepinephrine (MG)','Norepinephrine (MCG/KG/MIN)_Rate', 'Epinephrine (MG)','Epinephrine (MCG/KG/MIN)_Rate', 'Dopamine (MG)',
                'Dopamine (MCG/KG/MIN)_Rate', 'Dobutamine (MG)','Dobutamine (MCG/KG/MIN)_Rate', 'Phenylephrine (MG)','Phenylephrine (MCG/KG/MIN)_Rate',
                'Vasopressin (MG)','Vasopressin (MCG/KG/MIN)_Rate', 'Milrinone (MG)', 'ECMO_fillna'], axis = 1)

    mimic = mimic.drop(['Phenylephrine (MG)','Dopamine (MG)', 'Norepinephrine (MG)', 'Epinephrine (MG)',
    'Dobutamine (MG)', 'Nitroprusside (MG)', 'Milrinone (MG)','Phenylephrine (MCG/KG/MIN)_Rate',
    'Dopamine (MCG/KG/MIN)_Rate','Norepinephrine (MCG/KG/MIN)_Rate', 'Epinephrine (MCG/KG/MIN)_Rate',
    'Dobutamine (MCG/KG/MIN)_Rate', 'Nitroprusside (MCG/KG/MIN)_Rate','Milrinone (MCG/KG/MIN)_Rate',
    'Vasopressin (MG)','ECMO_fillna','Catheter_fillna', 'EKG_fillna', 'CXR_fillna', 'MRI_fillna',
    'IABP_fillna', 'Impella_fillna', 'Arrhythmia','Blood culture_fillna', 'EKG','Urine output_fillna',
    'Acute_Myocardial_Infarction', 'Adv__Valvular_HD', 'End-Stage_Cardiomyopathy', 'Myocarditis',
    'Pulmonary_Embolism', 'Tension_Pneumothorax', 'hypovolemia_external', 'hypovolemia_internal',
    'spinal_cord_injury','suspected_infection'], axis = 1)

    mimic['Mechanical_circ_support'] = mimic['Impella'] + mimic['IABP'] + mimic['ECMO']
    mimic['Mechanical_circ_support'] = mimic['Mechanical_circ_support'].apply(lambda x: 1 if not pd.isna(x) and x > 0 else 0)

    eicu['Mechanical_circ_support'] = eicu['IABP'] + eicu['ECMO']
    eicu['Mechanical_circ_support'] = eicu['Mechanical_circ_support'].apply(lambda x: 1 if not pd.isna(x) and x > 0 else 0)

    mimic = mimic.drop(['ECMO', 'IABP', 'Impella'], axis =1)
    eicu = eicu.drop(['ECMO', 'IABP'], axis =1)
    
    #컬럼 순서 맞추기
    
    # 두 데이터 프레임에서 공통으로 존재하는 컬럼
    common_columns = set(mimic.columns).intersection(set(eicu.columns))

    # 두 데이터 프레임에서 각각만 존재하는 컬럼
    unique_columns_mimic = set(mimic.columns) - common_columns
    unique_columns_eicu = set(eicu.columns) - common_columns

    # 컬럼 순서를 동일하게 맞추기
    # 공통 컬럼 + mimic의 고유 컬럼 + eicu의 고유 컬럼
    new_column_order = list(common_columns) + list(unique_columns_mimic) + list(unique_columns_eicu)

    # 두 데이터 프레임의 컬럼 순서를 새로운 순서로 재배치
    mimic = mimic.reindex(columns=new_column_order)
    eicu = eicu.reindex(columns=new_column_order)
    
    idx = mimic[mimic['Temperature']<0].index
    mimic['Temperature'].loc[idx] = 0
    mimic['Temperature_fillna'].loc[idx] = 1
    
    # Troponin 컬럼명 변경
    mimic = mimic.rename(columns = {'Tropinin-T' : 'Troponin-T','Tropinin-T_fillna' : 'Troponin-T_fillna'})
    eicu = eicu.rename(columns = {'Tropinin-T' : 'Troponin-T','Tropinin-T_fillna' : 'Troponin-T_fillna'})

    # mimic inf 값 처리
    mimic = replace_inf(mimic,'PaO2/FiO2','zero')
    
    return mimic.dropna(axis=1), eicu.dropna(axis=1)

if __name__ == "__main__":
    print('Dataset Preprocess...')
    mimic, eicu = integration_source_target(mimic_path,eicu_path)
    print(f'MIMIC Nums of Columns {len(mimic.columns)} | eICU Nums of Columns {len(eicu.columns)}')
    mimic.to_csv('./dataset/mimic_ARDS.csv',index = False)
    eicu.to_csv('./dataset/eicu_ARDS.csv',index = False)