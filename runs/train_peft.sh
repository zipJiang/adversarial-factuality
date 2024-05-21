#!/bin/bash
#SBATCH --job-name=new-train-peft
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=ba100

export SPLIT=$1
export MODEL_TYPE=$2
export PYTHONPATH=$(pwd):$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64/:/usr/local/cuda-12.1/lib64/:/brtx/602-nvme2/zpjiang/miniconda3/envs/train-llm/lib/:${LD_LIBRARY_PATH}

echo "Split: $SPLIT"

conda run -n train-llm --no-capture-output \
    python3 scripts/run_task.py configs/training/train_peft_${MODEL_TYPE}.yaml