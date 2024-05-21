#!/bin/bash
#SBATCH --job-name=new-train-unli
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6-dev

split=$1


export CACHE_PATH=.cache/.mistral-7b-cache.db
export SCORE_DIR=data/scores/tuned_generation/

export OUTPUT_PATH="data/tuned_generation/${split}-mistral.jsonl"
export SCORE_PATH="${SCORE_DIR}${split}-factscore.json"
conda run -p .env --no-capture-output \
    python scripts/run_task.py configs/factscore_configs.yaml \
    --cache-path $CACHE_PATH

export SCORE_PATH="${SCORE_DIR}${split}-dedup.json"
conda run -p .env --no-capture-output \
    python scripts/run_task.py configs/dedupsoft_configs.yaml \
    --cache-path $CACHE_PATH

# export OUTPUT_PATH=data/outputs_jack_first_dpo/post-dpo.json
# export SCORE_PATH=${SCORE_DIR}jack-post-dpo-full.json
# conda run -p .env --no-capture-output \
#     python scripts/run_task.py configs/dedupsoft_configs.yaml \
#     --cache-path $CACHE_PATH

# export SCORE_PATH=${SCORE_DIR}mistral-factscore.json
# conda run -p .env --no-capture-output \
#     python scripts/run_task.py configs/factscore_configs.yaml \
#     --cache-path $CACHE_PATH

# export SCORE_PATH=${SCORE_DIR}gpt3.5-factscore.json
# export CACHE_PATH=.cache/.gpt-3.5-turbo-cache.db
# conda run -p .env --no-capture-output \
#     python scripts/run_task.py configs/factscore_gpt3.5_configs.yaml \
#     --cache-path $CACHE_PATH

# export SCORE_PATH=${SCORE_DIR}gpt3.5-factscore-abstention-detection.json
# export CACHE_PATH=.cache/.gpt-3.5-turbo-cache.db
# conda run -p .env --no-capture-output \
#     python scripts/run_task.py configs/factscore_gpt3.5_configs.yaml \
#     --cache-path $CACHE_PATH