import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
from tqdm import tqdm
import random

warnings.filterwarnings('ignore')

def data_split(df, seed, stayid, mode):

    ## patientunitstayid를 기준으로 split 진행
    random.seed(seed)

    all_list = df[stayid].unique().tolist()

    if mode.startswith('s'): 
        sample_size = int(len(all_list) * 0.7)
    else:
        sample_size = int(len(all_list) * 0.3)
    
    train_unitid = random.sample(all_list,sample_size)
    df_train, df_valid  = df.query('@train_unitid in ' + stayid), df.query('@train_unitid not in ' + stayid)

    return df_train, df_valid


class TableDataset(Dataset):
    def __init__(self,data_path,data_type,mode,seed):
        self.data_path = data_path
        self.data_type = data_type # eicu or mimic
        self.mode = mode # s_train / s_test / t_train / t_test
        self.target = 'ARDS_next_12h'
        self.seed = seed

        self.cat_features = ['Mechanical_circ_support', 'CXR', 'MRI', 'Blood culture', 'Catheter',
                'Ventilator', 'gender', 'ethnicity', 'Antibiotics', 'vasoactive/inotropic', 'SpO2_fillna', 
                'Cardiac Output_fillna', 'ABPs_fillna', 'WBC_fillna', 'Glucose_fillna', 'Respiratory Rate_fillna',
                'HR_fillna', 'Temperature_fillna', 'Hemoglobin_fillna', 'AST_fillna', 'Creatinine_fillna', 'ABPd_fillna', 
                'CVP_fillna', 'PaO2_fillna', 'EtCO2_fillna', 'Eye Opening_fillna', 'RedBloodCell_fillna', 'Total Bilirubin_fillna', 
                'RASS_fillna', 'Troponin-T_fillna', 'NIBPd_fillna', 'Anion gap_fillna', 'NIBPs_fillna', 'pH_fillna', 'Potassium_fillna', 
                'Hematocrit_fillna','MAP_fillna', 'Platelets_fillna', 'height_fillna', 'Verbal Response_fillna', 'SVO2_fillna', 'Ca+_fillna', 
                'FIO2 (%)_fillna', 'O2 Sat (%)_fillna', 'Motor Response_fillna', 'PaCO2_fillna', 'INR_fillna', 'Alkaline phosphatase_fillna', 
                'PEEP_fillna', 'Peak Insp. Pressure_fillna', 'CO2_fillna', 'BUN_fillna', 'Stroke Volume_fillna', 'Lactate_fillna', 'CRP_fillna', 
                'Sodium_fillna', 'ALT_fillna']
        
        self.num_features = ['Time_since_ICU_admission', 'Urine output', 'NIBPs', 'O2 Sat (%)', 'Respiratory Rate',
                'HR', 'MAP', 'NIBPd', 'Temperature', 'RASS', 'ABPs', 'PEEP', 'Peak Insp. Pressure',
                'ABPd', 'CVP', 'Verbal Response', 'Motor Response', 'EtCO2', 'Eye Opening', 'SVO2',
                'Stroke Volume', 'Cardiac Output', 'SpO2', 'Lactate', 'Sodium', 'Creatinine', 'Anion gap',
                'Glucose', 'Potassium', 'pH', 'PaCO2', 'Ca+', 'Hemoglobin', 'PaO2', 'Alkaline phosphatase',
                'BUN', 'ALT', 'AST', 'FIO2 (%)', 'CO2', 'INR', 'Total Bilirubin', 'RedBloodCell', 'Hematocrit',
                'Platelets', 'WBC', 'Troponin-T', 'CRP', 'height', 'weight', 'Age', 'PaO2/FiO2', 'Fluids(ml)',
                'GCS_score', 'Sofa_Respiration', 'Sofa_Coagulation', 'Sofa_Liver', 'Sofa_Cardiovascular',
                'Sofa_GCS', 'Sofa_Urine', 'SoFa_score']
        
        self.df_num, self.df_cat, self.y = self.__prepare_data__()

    def __prepare_data__(self):
        df_raw = pd.read_csv(self.data_path)

        scaler = StandardScaler()

        # if dataset is mimic (source)
        if self.data_type == 'mimic':
            df_train, df_valid = data_split(df_raw,self.seed,'stay_id',self.mode)

            if self.mode == "s_train":
                X_num = df_train[self.num_features]
                X_num_scaled = scaler.fit_transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features]
                y = df_train[self.target]
                return X_num, X_cat, y
            
             # s_test mode
            else:
                X_num_standard = df_train[self.num_features]
                scaler.fit(X_num_standard)
                
                X_num = df_valid[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features]
                y = df_valid[self.target]
                return X_num, X_cat, y

        # if dataset is eicu (target)
        else:
            df_train, df_valid = data_split(df_raw,self.seed,'patientunitstayid', self.mode)
            df_scaling = pd.read_csv('./dataset/mimic_ARDS_cutoff.csv')
            df_scaling, _ = data_split(df_scaling,self.seed,'stay_id', 's_train')

            if self.mode == "t_train":
                X_num_standard = df_scaling[self.num_features]
                scaler.fit(X_num_standard)

                X_num = df_train[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_train[self.cat_features]
                y = df_train[self.target]
                return X_num, X_cat, y
            
            # t_test mode
            else : 
                ## scaler fitting을 위한 과정
                X_num_standard = df_scaling[self.num_features]
                scaler.fit(X_num_standard)

                X_num = df_valid[self.num_features]
                X_num_scaled = scaler.transform(X_num)
                X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
                X_cat = df_valid[self.cat_features]
                y = df_valid[self.target]
                
                return X_num, X_cat, y

    def __getitem__(self,index):
                
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32).long()

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]


class VisDataset(Dataset):
    def __init__(self,data_path,seed,label_type):
        self.data_path = data_path
        self.target = 'ARDS_next_12h'
        self.seed = seed

        self.cat_features = ['Mechanical_circ_support', 'CXR', 'MRI', 'Blood culture', 'Catheter',
                'Ventilator', 'gender', 'ethnicity', 'Antibiotics', 'vasoactive/inotropic', 'SpO2_fillna', 
                'Cardiac Output_fillna', 'ABPs_fillna', 'WBC_fillna', 'Glucose_fillna', 'Respiratory Rate_fillna',
                'HR_fillna', 'Temperature_fillna', 'Hemoglobin_fillna', 'AST_fillna', 'Creatinine_fillna', 'ABPd_fillna', 
                'CVP_fillna', 'PaO2_fillna', 'EtCO2_fillna', 'Eye Opening_fillna', 'RedBloodCell_fillna', 'Total Bilirubin_fillna', 
                'RASS_fillna', 'Troponin-T_fillna', 'NIBPd_fillna', 'Anion gap_fillna', 'NIBPs_fillna', 'pH_fillna', 'Potassium_fillna', 
                'Hematocrit_fillna','MAP_fillna', 'Platelets_fillna', 'height_fillna', 'Verbal Response_fillna', 'SVO2_fillna', 'Ca+_fillna', 
                'FIO2 (%)_fillna', 'O2 Sat (%)_fillna', 'Motor Response_fillna', 'PaCO2_fillna', 'INR_fillna', 'Alkaline phosphatase_fillna', 
                'PEEP_fillna', 'Peak Insp. Pressure_fillna', 'CO2_fillna', 'BUN_fillna', 'Stroke Volume_fillna', 'Lactate_fillna', 'CRP_fillna', 
                'Sodium_fillna', 'ALT_fillna']
        
        self.num_features = ['Time_since_ICU_admission', 'Urine output', 'NIBPs', 'O2 Sat (%)', 'Respiratory Rate',
                'HR', 'MAP', 'NIBPd', 'Temperature', 'RASS', 'ABPs', 'PEEP', 'Peak Insp. Pressure',
                'ABPd', 'CVP', 'Verbal Response', 'Motor Response', 'EtCO2', 'Eye Opening', 'SVO2',
                'Stroke Volume', 'Cardiac Output', 'SpO2', 'Lactate', 'Sodium', 'Creatinine', 'Anion gap',
                'Glucose', 'Potassium', 'pH', 'PaCO2', 'Ca+', 'Hemoglobin', 'PaO2', 'Alkaline phosphatase',
                'BUN', 'ALT', 'AST', 'FIO2 (%)', 'CO2', 'INR', 'Total Bilirubin', 'RedBloodCell', 'Hematocrit',
                'Platelets', 'WBC', 'Troponin-T', 'CRP', 'height', 'weight', 'Age', 'PaO2/FiO2', 'Fluids(ml)',
                'GCS_score', 'Sofa_Respiration', 'Sofa_Coagulation', 'Sofa_Liver', 'Sofa_Cardiovascular',
                'Sofa_GCS', 'Sofa_Urine', 'SoFa_score']

        self.df_raw, self.df_num, self.df_cat, self.y = self.__prepare_data__(label_type)

    def __prepare_data__(self,label_type):
        df_raw = pd.read_csv(self.data_path)
        # _, df_raw = data_split(df_raw,self.seed,'patientunitstayid', 't_test')
        _, df_raw = data_split(df_raw,self.seed,'stay_id', 's_train')
        
        if label_type == "positive":
            df_raw = df_raw[df_raw[self.target] == 1]
        else:
            df_raw = df_raw[df_raw[self.target] == 0]

        scaler = StandardScaler()

        ## scaler fitting을 위한 과정
        df_scaling = pd.read_csv('./dataset/mimic_ARDS_cutoff.csv')
        df_scaling, _ = data_split(df_scaling,self.seed,'stay_id','s_train')
        X_num_standard = df_scaling[self.num_features]
        scaler.fit(X_num_standard)

        X_num = df_raw[self.num_features]
        X_num_scaled = scaler.transform(X_num)
        X_num = pd.DataFrame(X_num_scaled,columns = X_num.columns)
        X_cat = df_raw[self.cat_features]
        y = df_raw[self.target]
        
        return df_raw, X_num, X_cat, y

    def __getitem__(self,index):
                
        X_num_features = torch.tensor(self.df_num.iloc[index,:].values,dtype=torch.float32)
        X_cat_features = torch.tensor(self.df_cat.iloc[index,:].values,dtype=torch.float32).long()
        label = torch.tensor(int(self.y.iloc[index]),dtype=torch.float32).long()

        return X_num_features, X_cat_features, label
    
    def __len__(self):
        return self.y.shape[0]