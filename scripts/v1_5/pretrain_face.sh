
JSON_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/facellava/train_json"
IMAGE_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/facellava"
VIDEO_FOLDER="/wekafs/ict/achaubey/emotion_reasoning/data/facellava"

# cd /path/to/Video-LLaVA
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed facellava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /wekafs/ict/achaubey/emotion_reasoning/code/data_preprocess/instruct_files/mafw_dfew_ferv39k_balanced_12k.json \
    --image_folder /wekafs/ict/achaubey/emotion_reasoning/data \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder /wekafs/ict/achaubey/emotion_reasoning/data \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type naive_face_mask_adapter \
    --pretrain_mm_mlp_adapter ./checkpoints/facellava-7b-pretrain_attn/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/facellava-7b-pretrain_face_attn \
    --num_train_epochs 1 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
