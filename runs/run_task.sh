#!/bin/bash
#SBATCH --job-name=run_task
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zjiang31
#SBATCH --partition=a100

export PYTHONPATH=$(pwd):$PYTHONPATH
# export CUDA_LAUNCH_BLOCKING=1
# export HF_HOME=$(pwd)/data/hub-home/
export HF_HOME=/scratch/bvandur1/zjiang31/conformal-backoff/data/hub-home/
export PYSERINI_CACHE=/scratch/bvandur1/zjiang31/adversarial-factuality/.cache/pyserini

declare -a CONFIG_PATHS
CONFIG_PATHS=()
EXCLUDE_LAST=0
USE_ACCELERATE=0

# set default values
SERVE_VLLM=0
GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))

cleanup() {
    echo "Caught signal, cleaning up..."
    # kill vllm server
    if [ $SERVE_VLLM -eq 1 ]; then
        pkill -f "vllm serve"
    fi
    # also need to kill all background processes
    kill $(jobs -p)
    exit 1
}

trap cleanup SIGINT
trap cleanup SIGTERM
trap cleanup SIGKILL
trap cleanup EXIT

# print nvidia-smi
nvidia-smi

if [[ ${#GPUS[@]} -ge 1 ]]; then
    LAST_GPU=${GPUS[-1]}
    echo "GPUs available: ${GPUS[@]}"
    echo "Last GPU set to ${LAST_GPU}"
else
    LAST_GPU=
    echo "No GPUs available"
fi

# parse --argument
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    --config-path)
    # CONFIG_PATH="$2"
    CONFIG_PATHS+=("$2")
    shift
    shift
    ;;
    --serve-vllm)
    SERVE_VLLM=1
    # if [[ ${#GPUS[@]} -lt 1 ]]; then
    #     echo "Error: Less than 1 GPUs available."
    #     exit 1
    # fi
    shift
    ;;
    --model-name)
    MODEL_NAME="$2"
    shift
    shift
    ;;
    --use-accelerate)
    USE_ACCELERATE=1
    shift
    ;;
    --exclude-last)
    EXCLUDE_LAST=1
    shift
    ;;
    *)
    shift
    ;;
esac
done

# conda run -p .env --no-capture-output \
#     python scripts/test_gpu.py

echo "CONFIG_PATHS to run:"
echo "-------------------"
for config_path in "${CONFIG_PATHS[@]}"
do
    echo $config_path
done
echo "-------------------"

if [ $SERVE_VLLM -eq 1 ]; then
    # serve vllm in the background with log directed to vllm_log.txt
    # use the n - 1 first GPUs
    if [ $EXCLUDE_LAST -eq 1 ]; then
        selected_gpus=${GPUS[@]::${#GPUS[@]}-1}
    else
        selected_gpus=${GPUS[@]}
    fi
    num_gpus=$(echo $selected_gpus | wc -w)

    rm -f vllm_log.txt

    # if model_name is not provided, use the default model
    if [ -z ${MODEL_NAME} ]; then
        CUDA_VISIBLE_DEVICES=${selected_gpus// /,} runs/serve.sh --tp-size ${num_gpus} > vllm_log.txt 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=${selected_gpus// /,} runs/serve.sh --model-name ${MODEL_NAME} --tp-size ${num_gpus} > vllm_log.txt 2>&1 &
    fi

    while true; do
        if grep -q "Avg prompt throughput" vllm_log.txt; then
            break
        fi
        sleep 10
    done
fi


for config_path in "${CONFIG_PATHS[@]}"
do
    if [ $USE_ACCELERATE -eq 1 ]; then
        conda run -p .env --no-capture-output \
            accelerate launch \
                --multi_gpu \
                --num_processes=4 \
                scripts/run_task.py --config-path ${config_path}
    else
        CUDA_VISIBLE_DEVICES=${LAST_GPU} conda run -p .env --no-capture-output \
            python scripts/run_task.py --config-path ${config_path}
    fi
done

# kill vllm server
# if [ $SERVE_VLLM -eq 1 ]; then
#     pkill -f "vllm serve"
# fi

# # also need to kill all background processes
# trap 'kill $(jobs -p)' EXIT
