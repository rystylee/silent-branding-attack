export CUDA_VISIBLE_DEVICES="1"

export DATASET_METADATA_PATH="dataset/midjourney/metadata.jsonl"
export REF_IMAGES_PATH="dataset/unlabeled_test/unlabeled_refs"
export LORA_PATH="output/logo_personalization/unlabeled_test/save-3000"
export OUT_DIR="output/poisoned_images/unlabeled_test"
export TOK="'U' logo"


python generate_poisoned_images.py \
  --dataset_metadata_path=$DATASET_METADATA_PATH \
  --tok="$TOK" \
  --ref_images_path=$REF_IMAGES_PATH \
  --lora_path=$LORA_PATH \
  --out_dir=$OUT_DIR
