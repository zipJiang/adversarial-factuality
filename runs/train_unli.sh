#!/bin/bash
#SBATCH --job-name=new-train-unli
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6


export PYTHONPATH=$(pwd):$PYTHONPATH
export OUTPUT_PATH=ckpt/tuned_unli

conda run -p .env --no-capture-output \
    python3 scripts/run_task.py configs/training/train_unli.yaml