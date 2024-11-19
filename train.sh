# Note! 
# You will need to change the 3 dirs below for each experiment attempt.

# ## cutoff_1 , 2는 focal loss 3, 4에서 모두 nn.CrossEntrtopy를 사용한 것임.
# python main.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 10\
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/final_DA_Strict" \
#     --log_dir "./log/final_DA_Strict" \
#     --result_dir "./result/final_DA_Strict" \
#     --mode "train"

# python main_uni.py \
#     --lr 0.0001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 10\
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/final_uni_focal_Strict" \
#     --log_dir "./log/final_uni_focal_Strict" \
#     --result_dir "./result/final_uni_focal_Strict" \
#     --mode "train"
    
# python main.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --dim 128 \
#     --dim_head 64 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/cut_off_4" \
#     --log_dir "./log/cut_off_4" \
#     --result_dir "./result/cut_off_4" \
#     --mode "train"

# ########## HIRID & PIC Dataset ##########
# python main.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_picu" \
#     --log_dir "./log/hirid_picu" \
#     --result_dir "./result/hirid_picu" \
#     --mode "train"

# python main_uni.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_picu_uni" \
#     --log_dir "./log/hirid_picu_uni" \
#     --result_dir "./result/hirid_picu_uni" \
#     --mode "train"

########## HIRID & MIMIC Dataset ##########
# python main.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic" \
#     --log_dir "./log/hirid_mimic" \
#     --result_dir "./result/hirid_mimic" \
#     --mode "train" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 

# python main_uni.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_uni" \
#     --log_dir "./log/hirid_mimic_uni" \
#     --result_dir "./result/hirid_mimic_uni" \
#     --mode "train" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 


# ########## HIRID & PIC Dataset ##########
# python main.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_picu_drop" \
#     --log_dir "./log/hirid_picu_drop" \
#     --result_dir "./result/hirid_picu_drop" \
#     --mode "train"

# python main_uni.py \
#     --lr 0.001\
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 100 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 100\
#     --num_cont 49 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_picu_uni_drop" \
#     --log_dir "./log/hirid_picu_uni_drop" \
#     --result_dir "./result/hirid_picu_uni_drop" \
#     --mode "train"

########## HIRID & MIMIC Split Dataset Drop Derivate Variable ##########
# python main.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 500 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 500\
#     --num_cont 22 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_drop" \
#     --log_dir "./log/hirid_mimic_drop" \
#     --result_dir "./result/hirid_mimic_drop" \
#     --mode "train" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 

# python main_uni.py \
#     --lr 0.001 \
#     --weight_decay 0.00001 \
#     --batch_size 256 \
#     --num_epoch 500 \
#     --T_max 25 \
#     --scheduler \
#     --early_stop \
#     --patience 500\
#     --num_cont 22 \
#     --num_cat 9 \
#     --dim 32 \
#     --dim_head 16 \
#     --depth 6 \
#     --heads 8 \
#     --attn_dropout 0.1 \
#     --ff_dropout 0.1 \
#     --ckpt_dir "./checkpoint/hirid_mimic_uni_drop" \
#     --log_dir "./log/hirid_mimic_uni_drop" \
#     --result_dir "./result/hirid_mimic_uni_drop" \
#     --mode "train" \
#     --pic_data_dir "./dataset/MIMIC_ARDS_12H_SPLIT.csv.gz" 

########## HIRID & MIMIC All Dataset  ##########
python main.py \
    --lr 0.001 \
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 20\
    --num_cont 49 \
    --num_cat 9 \
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/hirid_mimic_all" \
    --log_dir "./log/hirid_mimic_all" \
    --result_dir "./result/hirid_mimic_all" \
    --mode "train" \
    --hirid_data_dir "./dataset/HIRID_ARDS_12H.csv.gz" \
    --pic_data_dir "./dataset/MIMIC_ARDS_12H.csv.gz" 

python main_uni.py \
    --lr 0.001 \
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 20\
    --num_cont 49 \
    --num_cat 9 \
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/hirid_mimic_uni_all" \
    --log_dir "./log/hirid_mimic_uni_all" \
    --result_dir "./result/hirid_mimic_uni_all" \
    --mode "train" \
    --hirid_data_dir "./dataset/HIRID_ARDS_12H.csv.gz" \
    --pic_data_dir "./dataset/MIMIC_ARDS_12H.csv.gz" 
