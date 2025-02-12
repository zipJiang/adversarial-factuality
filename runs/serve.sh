#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=ba100
#SBATCH --gres=gpu:1


export PROJECT_DIR=/scratch/bvandur1/zjiang31/conformal-backoff/
export TP_SIZE=1
export PORT=9871
export MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --model-name)
    export MODEL_NAME="$2"
    shift
    shift
    ;;
    --port)
    export PORT="$2"
    shift
    shift
    ;;
    --tp-size)
    export TP_SIZE="$2"
    shift
    shift
    ;;
    *)
    shift
    ;;
esac
done

echo "--- Environment Variables ---"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "MODEL_NAME: $MODEL_NAME"
echo "PORT: $PORT"
echo "TP_SIZE: $TP_SIZE"
echo "--- Environment Variables ---"

conda run --no-capture-output -p ${PROJECT_DIR}.env-vllm \
    vllm serve ${MODEL_NAME} \
        --dtype auto \
        --tensor-parallel-size ${TP_SIZE} \
        --api-key "token-abc123" \
        --download-dir ${PROJECT_DIR}/data/models \
        --port ${PORT}
