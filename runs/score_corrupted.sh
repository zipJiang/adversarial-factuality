#!/bin/bash
#SBATCH --job-name=score-corrupted
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6

export PARENT_DIR=/brtx/606-nvme2/zpjiang/adversarial-factuality/
export PYTHONPATH="${PARENT_DIR}:${PYTHONPATH}"
export CACHE_PATH=${PARENT_DIR}.cache/.mistral-7b-cache.db
export SCORE_DIR=${PARENT_DIR}data/scores/corrupted/

export GENERATION_DIR="${PARENT_DIR}data/corrupted/"

"/brtx/606-nvme2/zpjiang/adversarial-factuality/runs/port_binding.sh"

# makedir score_dir
mkdir -p $SCORE_DIR

for filepath in ${GENERATION_DIR}*.jsonl; do
    export OUTPUT_PATH=$filepath
    export SCORE_PATH="${SCORE_DIR}$(basename $filepath .jsonl)-factscore.json"
    conda run -p ${PARENT_DIR}.env --no-capture-output \
        python ${PARENT_DIR}scripts/run_task.py ${PARENT_DIR}configs/factscore_configs.yaml \
        --cache-path $CACHE_PATH

    export SCORE_PATH="${SCORE_DIR}$(basename $filepath .jsonl)-dedup.json"
    conda run -p ${PARENT_DIR}.env --no-capture-output \
        python ${PARENT_DIR}scripts/run_task.py ${PARENT_DIR}configs/dedupsoft_configs.yaml \
        --cache-path $CACHE_PATH
done