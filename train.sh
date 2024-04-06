# Note! 
# You will need to change the 3 dirs below for each experiment attempt.

## cutoff_1 , 2는 focal loss 3, 4에서 모두 nn.CrossEntrtopy를 사용한 것임.
python main.py \
    --lr 0.001\
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 10\
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/final_DA_Strict" \
    --log_dir "./log/final_DA_Strict" \
    --result_dir "./result/final_DA_Strict" \
    --mode "train"

python main_uni.py \
    --lr 0.0001\
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 10\
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/final_uni_focal_Strict" \
    --log_dir "./log/final_uni_focal_Strict" \
    --result_dir "./result/final_uni_focal_Strict" \
    --mode "train"
    
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
