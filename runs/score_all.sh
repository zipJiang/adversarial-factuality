#!/bin/bash
#SBATCH --job-name=new-train-unli
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6

split=$1
model_type=$2

export PYTHONPATH="/brtx/606-nvme2/zpjiang/adversarial-factuality:${PYTHONPATH}"
export CACHE_PATH=.cache/.mistral-7b-cache.db
export SCORE_DIR=data/scores/tuned_generation/

export OUTPUT_PATH="data/tuned_generation/${split}-${model_type}.jsonl"
export SCORE_PATH="${SCORE_DIR}${split}-${model_type}-factscore.json"
conda run -p .env --no-capture-output \
    python scripts/run_task.py configs/factscore_configs.yaml \
    --cache-path $CACHE_PATH

export SCORE_PATH="${SCORE_DIR}${split}-${model_type}-dedup.json"
conda run -p .env --no-capture-output \
    python scripts/run_task.py configs/dedupsoft_configs.yaml \
    --cache-path $CACHE_PATH