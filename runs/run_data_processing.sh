#!/bin/bash
#SBATCH --job-name=preprocess-
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --partition=brtx6


export PYTHONPATH=$(pwd):$PYTHONPATH
export ENV_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/.env

conda run -p $ENV_PATH --no-capture-output \
    python3 scripts/run_task.py configs/preprocessing/filter_required_entities.yaml