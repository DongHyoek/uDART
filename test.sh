# Note! 
# For each experimental attempt, you need to change the 3 dirs below. And the rest of the parsers, except for mode, should be the same as in the train.sh file.
python main.py \
    --lr 0.001\
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 15\
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/final_DA_Strict" \
    --log_dir "./log/final_DA_Strict" \
    --result_dir "./result/final_DA_Strict" \
    --mode "test"

python main_uni.py \
    --lr 0.001\
    --weight_decay 0.00001 \
    --batch_size 256 \
    --num_epoch 100 \
    --T_max 25 \
    --scheduler \
    --early_stop \
    --patience 15\
    --dim 32 \
    --dim_head 16 \
    --depth 6 \
    --heads 8 \
    --attn_dropout 0.1 \
    --ff_dropout 0.1 \
    --ckpt_dir "./checkpoint/final_uni_focal_Strict" \
    --log_dir "./log/final_uni_focal_Strict" \
    --result_dir "./result/final_uni_focal_Strict" \
    --mode "test"