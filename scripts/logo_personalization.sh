## IMPORTANT: overfitted weights (3000 steps) show better inpainting (editing) results

export CUDA_VISIBLE_DEVICES="1"
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"


# # midjourney
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# export TRAIN_CONFIG_PATH="dataset/logo_example/avengers/metadata.jsonl"
# export REG_CONFIG_PATH="dataset/midjourney/metadata.jsonl"
# export WANDB_NAME="logo_personalization_avengers_test"
# export OUTPUT_DIR="output/logo_personalization/avengers_test"
# export MAX_TRAIN_STEPS=3010

# accelerate launch --config_file config/default.yaml logo_personalization_sdxl.py \
#   --pretrained_model_name_or_path=$MODEL_NAME   --pretrained_vae_model_name_or_path=$VAE_NAME \
#   --mixed_precision="fp16" \
#   --train_config_path=$TRAIN_CONFIG_PATH --reg_config_path=$REG_CONFIG_PATH --with_prior_preservation --caption_column="text" \
#   --resolution=1024 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 --rank=256 \
#   --max_train_steps=$MAX_TRAIN_STEPS --checkpointing_steps=100010 --save_steps=1000 --validation_epochs 10 \
#   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --seed=42 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a 'A' logo backpack on sunny hill, 4K, high quality" --report_to="wandb" --wandb_name=$WANDB_NAME --gradient_checkpointing

# # tarot
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# export TRAIN_CONFIG_PATH="dataset/logo_example/avengers/metadata.jsonl"
# export REG_CONFIG_PATH="dataset/tarot/metadata.jsonl"
# export WANDB_NAME="logo_personalization_avengers_tarot_test"
# export OUTPUT_DIR="output/logo_personalization/avengers_tarot_test"
# export MAX_TRAIN_STEPS=3010

# accelerate launch --config_file config/default.yaml logo_personalization_sdxl.py \
#   --pretrained_model_name_or_path=$MODEL_NAME   --pretrained_vae_model_name_or_path=$VAE_NAME \
#   --mixed_precision="fp16" \
#   --train_config_path=$TRAIN_CONFIG_PATH --reg_config_path=$REG_CONFIG_PATH --with_prior_preservation --caption_column="text" \
#   --resolution=1024 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 --rank=256 \
#   --max_train_steps=$MAX_TRAIN_STEPS --checkpointing_steps=100010 --save_steps=1000 --validation_epochs 10 \
#   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --seed=42 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a trtcrd of 'A' logo backpack on sunny hill" --report_to="wandb" --wandb_name=$WANDB_NAME --gradient_checkpointing

# ghost in the shell
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export TRAIN_CONFIG_PATH="dataset/logo_example/avengers/metadata.jsonl"
export REG_CONFIG_PATH="dataset/ghost_in_the_shell/metadata.jsonl"
export WANDB_NAME="logo_personalization_avengers_ghost_in_the_shell"
export OUTPUT_DIR="output/logo_personalization/avengers_ghost_in_the_shell_step_9010"
export MAX_TRAIN_STEPS=9010

accelerate launch --config_file config/default.yaml logo_personalization_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME   --pretrained_vae_model_name_or_path=$VAE_NAME \
  --mixed_precision="fp16" \
  --train_config_path=$TRAIN_CONFIG_PATH --reg_config_path=$REG_CONFIG_PATH --with_prior_preservation --caption_column="text" \
  --resolution=1024 \
  --train_batch_size=1 --gradient_accumulation_steps=1 --rank=256 \
  --max_train_steps=$MAX_TRAIN_STEPS --checkpointing_steps=100010 --save_steps=1000 --validation_epochs 10 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="a 'A' logo backpack on sunny hill, 4K, high quality" --report_to="wandb" --wandb_name=$WANDB_NAME --gradient_checkpointing


# # unlabeled / midjourney
# export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
# export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
# export TRAIN_CONFIG_PATH="dataset/unlabeled_test/unlabeled/metadata.jsonl"
# export REG_CONFIG_PATH="dataset/midjourney/metadata.jsonl"
# export WANDB_NAME="logo_personalization_unlabeled_test_midjourney"
# export OUTPUT_DIR="output/logo_personalization/unlabeled_test_02"
# export MAX_TRAIN_STEPS=6010

# accelerate launch --config_file config/default.yaml logo_personalization_sdxl.py \
#   --pretrained_model_name_or_path=$MODEL_NAME   --pretrained_vae_model_name_or_path=$VAE_NAME \
#   --mixed_precision="fp16" \
#   --train_config_path=$TRAIN_CONFIG_PATH --reg_config_path=$REG_CONFIG_PATH --with_prior_preservation --caption_column="text" \
#   --resolution=1024 \
#   --train_batch_size=1 --gradient_accumulation_steps=1 --rank=256 \
#   --max_train_steps=$MAX_TRAIN_STEPS --checkpointing_steps=100010 --save_steps=1000 --validation_epochs 10 \
#   --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
#   --seed=42 \
#   --output_dir=$OUTPUT_DIR \
#   --validation_prompt="a 'U' logo backpack on sunny hill, 4K, high quality" --report_to="wandb" --wandb_name=$WANDB_NAME --gradient_checkpointing
