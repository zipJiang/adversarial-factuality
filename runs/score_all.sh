#!/bin/bash


export OUTPUT_PATH=data/outputs/generation_opt_corner_cases.json
export CACHE_PATH=.cache/.mistral-7b-cache.db
export SCORE_DIR=data/scores/local_models/

export SCORE_PATH=${SCORE_DIR}mistral-soft-deberta-base-parallel-corner.json
conda run -p .env --no-capture-output \
    python scripts/run_task.py configs/dedupsoft_configs.yaml \
    --cache-path $CACHE_PATH

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