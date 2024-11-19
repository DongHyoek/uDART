import argparse
import os
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import *
from data_provider import *

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Visualize Attention Map",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Data
parser.add_argument("--eicu_data_dir", default="./dataset/eicu_ARDS_cutoff.csv", type=str, dest="eicu_data_dir")
parser.add_argument("--mimic_data_dir", default="./dataset/mimic_ARDS_cutoff.csv", type=str, dest="mimic_data_dir")
parser.add_argument('--seed', default=42, type=int , dest='seed')
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--label_type", default='positive', type=str, dest="label_type", help = 'positive or negative')

# Model
parser.add_argument("--num_cont", default=61, type=int, dest="num_cont", help = "Nums of Continuous Features")
parser.add_argument("--num_cat", default=57, type=int, dest="num_cat", help = "Nums of Categorical Features But Not Use")
parser.add_argument("--dim", default=32, type=int, dest="dim", help = "Embedding Dimension of Input Data ")
parser.add_argument("--dim_head", default=16, type=int, dest="dim_head", help = "Dimension of Attention(Q,K,V)")
parser.add_argument("--depth", default=6, type=int, dest="depth", help = "Nums of Attention Layer Depth")
parser.add_argument("--heads", default=8, type=int, dest="heads", help='Nums of Attention head')
parser.add_argument("--attn_dropout", default=0.1, type=float, dest="attn_dropout", help='Ratio of Attention Layer dropout')
parser.add_argument("--ff_dropout", default=0.1, type=float, dest="ff_dropout", help='Ratio of FeedForward Layer dropout')

# Others
parser.add_argument("--ckpt_dir", default="./checkpoint/1st", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result/1st", type=str, dest="result_dir")


args = parser.parse_args()

def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

fix_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(f'{args.result_dir}/{args.label_type}'):
    os.makedirs(os.path.join(f'{args.result_dir}/{args.label_type}'))

## Build Dataset 
print(f'Build Visualize Dataset ....')

dataset_train = TableDataset(data_path=args.mimic_data_dir, data_type='mimic',mode='train',seed=args.seed)

# Tuple Containing the number of unique values within each category
card_categories = []
for col in dataset_train.df_cat.columns:
    card_categories.append(dataset_train.df_cat[col].nunique())

del dataset_train

dataset_vis = VisDataset(data_path=args.mimic_data_dir,seed=args.seed, label_type = args.label_type)
dataloader_vis = DataLoader(dataset_vis,batch_size=args.batch_size, shuffle=False, num_workers=8)

## Prepare Model
ft_transformer_config = {
    'categories' : card_categories,      
    'num_continuous' : args.num_cont,                
    'dim' : args.dim,                                           
    'depth' : args.depth,                         
    'heads' : args.heads, 
    'dim_head' : args.dim_head,                      
    'attn_dropout' : args.attn_dropout,              
    'ff_dropout' : args.ff_dropout  
}

model = DANN(
    dim_feat = args.dim,
    transformer_config = ft_transformer_config                 
).to(device)

print(f'Visualize AttentionMap....')
print(f'Checkpoint Load....')
model.load_state_dict(torch.load(f'{args.ckpt_dir}/Best_DANN_Transformer.pth')['model_state_dict'])

with torch.no_grad():
    model.eval()
    columns = ['CLS_Token'] + dataset_vis.df_cat.columns[:10].tolist() + dataset_vis.df_num.columns.tolist()    
    alpha = 0

    for idx, batch_data in enumerate(tqdm(dataloader_vis)):
        X_num, X_cat, label = batch_data
        X_num, X_cat, label = X_num.to(device), X_cat.to(device), label.to(device)
        # X_cat = torch.tensor(dataset_vis.df_cat.iloc[args.df_index,:],dtype=torch.float32).long().to(device)
        # X_num = torch.tensor(dataset_vis.df_num.iloc[args.df_index,:],dtype=torch.float32).to(device)
        attn_map, latent_vector ,class_output,_ = model(X_cat,X_num, True, alpha) # atttnmap -> (depth,b,head,seq,seq)
        
        pred = class_output.detach().max(1, keepdim=True)[1]
    
        if not idx:
            cum_attn_map = attn_map[:,:,:,0:1,:].detach().cpu().numpy() # 각 배치마다 평균 
            pred_list = pred.detach().cpu().numpy()
            latent_arrays = latent_vector.detach().cpu().numpy()
    
        else:
            cum_attn_map = np.hstack((cum_attn_map,attn_map[:,:,:,0:1,:].detach().cpu().numpy()))
            pred_list = np.vstack((pred_list,pred.detach().cpu().numpy()))
            latent_arrays = np.vstack((latent_arrays,latent_vector.detach().cpu().numpy()))

    # # delete_fillna_value
    # final_attn_map = cum_attn_map / len(dataloader_vis)
    # final_attn_map = np.delete(final_attn_map, range(11,58), axis = 2)
    # final_attn_map = np.delete(final_attn_map, range(11,58), axis = 3)

    # # result dataframe save
    # result_df = dataset_vis.df_raw[['uniquepid','patientunitstayid','Time_since_ICU_admission','Annotation','ARDS_next_12h']]
    result_df = dataset_vis.df_raw[['subject_id','stay_id','Time_since_ICU_admission','Annotation','ARDS_next_12h']]
    result_df['pred'] = pred_list
    # result_df = pd.concat([result_df, pd.DataFrame(latent_arrays).rename(columns = {i : f'latent_dim{i+1}'for i in range(32)})],axis = 1)
    result_df.to_csv(f'{args.result_dir}/{args.label_type}_mimic_result.csv',index = False)
    np.save(f'{args.result_dir}/{args.label_type}_mimic_latent_vector.npy',latent_arrays)

    # for layer in range(attn_map.shape[0]):
    #     # make Heatmap
    #     plt.figure(figsize=(35, 35))
    #     plt.title(f'Layer_{layer+1} Attention Map')
    #     sns.heatmap(np.mean(final_attn_map[layer,:,:,:],axis=0).squeeze(), cmap='coolwarm', 
    #                 xticklabels= columns, yticklabels = columns, linewidths = 0.1 )
    #     plt.xticks(rotation=90)
    #     plt.yticks(rotation=0)
    #     plt.savefig(args.result_dir +f'/{args.label_type}' +f'/Layer_{layer+1}_Attentionmap.png')

# get feature importance

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)



columns = ['CLS_Token'] + dataset_vis.df_cat.columns.tolist() + dataset_vis.df_num.columns.tolist()

valid_indices = [i for i, col in enumerate(columns) if "_fillna" not in col and col != "progress" and col != "CRP" and col != "CLS_Token"]

ps = []

for sample in range(len(dataloader_vis)):
    get_cls_attentionmap = cum_attn_map[:, sample,:, :, :]  # (l, h, 1, f), sample에는 for 문의 결과로 1개의 샘플이 들어감
    sigma = get_cls_attentionmap.sum(axis=0).sum(axis=0)
    p = sigma / (args.heads * args.depth)
    ps.append(p)


p = np.array(ps).sum(axis=0) / len(dataloader_vis) * 100

if p.ndim == 2 and p.shape[0] == 1:
    p = p.ravel()

p_filtered = p[valid_indices]
top_feature_indices = np.argsort(p_filtered)[-10:][::-1]
top_p_values = p_filtered[top_feature_indices]
top_feature_names = [columns[valid_indices[i]] for i in top_feature_indices]


plt.figure(figsize=(10, 5)) 
bars = plt.bar(range(10), top_p_values, color=plt.cm.viridis(np.linspace(0, 1, 10)), edgecolor='black')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, round(yval, 3), ha='center', va='bottom', fontsize=8)

plt.title(f'Feature Importance - Case {args.label_type}', fontsize=25, fontweight='bold')
plt.xticks(ticks=range(10), labels=top_feature_names, rotation=45, ha="right", fontsize=12)  # x축 레이블 개선
plt.ylabel('Importance', fontsize=18)
plt.xlabel('Features', fontsize=18)
plt.tight_layout()
plt.savefig(args.result_dir +f'/{args.label_type}' +f'/mimic_Feature_Importance_Attention_map.png')