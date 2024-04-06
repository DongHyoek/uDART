python visualize_map.py \
    --label_type 'positive' \
    --batch_size 512 \
    --ckpt_dir "./checkpoint/final_DA_Strict" \
    --result_dir "./result/final_DA_Strict" \
    --dim 32 \
    --dim_head 16 \

python visualize_map_uni.py \
    --label_type 'positive' \
    --batch_size 512 \
    --ckpt_dir "./checkpoint/final_uni_focal_Strict" \
    --result_dir "./result/final_uni_focal_Strict" \
    --dim 32 \
    --dim_head 16 \

python visualize_map.py \
    --label_type 'negative' \
    --batch_size 256 \
    --ckpt_dir "./checkpoint/final_DA_Strict" \
    --result_dir "./result/final_DA_Strict" \
    --dim 32 \
    --dim_head 16 \

python visualize_map_uni.py \
    --label_type 'negative' \
    --batch_size 256 \
    --ckpt_dir "./checkpoint/final_uni_focal_Strict" \
    --result_dir "./result/final_uni_focal_Strict" \
    --dim 32 \
    --dim_head 16 