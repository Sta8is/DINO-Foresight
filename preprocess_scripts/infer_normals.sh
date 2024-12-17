export CUDA=0

export CHECKPOINT_DIR="jingheya/lotus-normal-d-v1-0"
export MODE="regression"

for SPLIT in train val test; do
    export BASE_PATH="/path/to/cityscapes/leftImg8bit" # Change this path accordingly
    export TEST_IMAGES="${BASE_PATH}/${SPLIT}"
    export OUTPUT_DIR="output/normals_${SPLIT}"

    CUDA_VISIBLE_DEVICES=$CUDA python infer.py \
        --pretrained_model_name_or_path=$CHECKPOINT_DIR \
        --prediction_type="sample" \
        --seed=42 \
        --half_precision \
        --input_dir=$TEST_IMAGES \
        --task_name=$TASK_NAME \
        --mode=$MODE \
        --output_dir=$OUTPUT_DIR \
        --disparity
done
