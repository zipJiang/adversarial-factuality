#!/bin/bash
#SBATCH --job-name=score-all
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6

split=$1
model_type=$2

export PARENT_DIR=/brtx/606-nvme2/zpjiang/adversarial-factuality/
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
export CACHE_PATH=${PARENT_DIR}.cache/.mistral-7b-cache.db
export SCORE_DIR=${PARENT_DIR}data/scores/tuned_generation/

export OUTPUT_PATH="${PARENT_DIR}data/tuned_generation/${split}-${model_type}.jsonl"
export SCORE_PATH="${SCORE_DIR}${split}-${model_type}-factscore.json"

"/brtx/606-nvme2/zpjiang/adversarial-factuality/runs/port_binding.sh"

conda run -p ${PARENT_DIR}.env --no-capture-output \
    python ${PARENT_DIR}scripts/run_task.py ${PARENT_DIR}configs/factscore_configs.yaml \
    --cache-path $CACHE_PATH

export SCORE_PATH="${SCORE_DIR}${split}-${model_type}-dedup.json"
conda run -p ${PARENT_DIR}.env --no-capture-output \
    python ${PARENT_DIR}scripts/run_task.py ${PARENT_DIR}configs/dedupsoft_configs.yaml \
    --cache-path $CACHE_PATH