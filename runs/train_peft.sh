#!/bin/bash
#SBATCH --job-name=new-train-peft
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=ba100

export SPLIT=$1
export MODEL_TYPE=$2
export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64/:/usr/local/cuda-12.1/lib64/:/brtx/602-nvme2/zpjiang/miniconda3/envs/train-llm/lib/:${LD_LIBRARY_PATH}
export BASE_MODEL=$(grep "model_name" configs/training/train_peft_${MODEL_TYPE}.yaml | cut -d ":" -f 2 | tr -d '[:space:]' | tr -d '"' | tr -d "'")
# Replace ${SPLIT} with the value of SPLIT
export ADAPTOR_PATH=$(grep "output_dir" configs/training/train_peft_${MODEL_TYPE}.yaml | cut -d ":" -f 2 | tr -d '[:space:]' | tr -d '"' | tr -d "'" | sed "s/\${SPLIT}/${SPLIT}/g")
export ONAME="${SPLIT}-${MODEL_TYPE}"

echo "Split: $SPLIT"

# If the model is trained, remove and re-train (all results will be saved in `wandb`)
if [ -d $ADAPTOR_PATH ]; then
    echo "Adaptor Model exists, removing..."
    rm -rf $ADAPTOR_PATH
fi

if [ -d ckpt/servables/${ONAME} ]; then
    echo "Merged Model exists, removing..."
    rm -rf ckpt/servables/${ONAME}
fi

conda run -n train-llm --no-capture-output \
    python3 scripts/run_task.py configs/training/train_peft_${MODEL_TYPE}.yaml

echo "Base model: $BASE_MODEL"
echo "Adaptor path: $ADAPTOR_PATH"

conda run -n train-llm --no-capture-output \
    python3 scripts/run_task.py configs/misc/merge_lora.yaml

# then, run the generation task to get local generation.

conda run -n train-llm --no-capture-output \
    python3 scripts/run_task.py configs/local_generation_configs.yaml