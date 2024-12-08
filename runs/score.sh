#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=brtx6


runs/port_binding.sh
export PYTHONPATH=/brtx/606-nvme2/zpjiang/adversarial-factuality:$PYTHONPATH
export SERPER_API_KEY="1f712f42c907b129ce68162ee1eaf6fb9d0c3be1"

MODEL_NAME=$1
CORRUPTION=$2
SPLIT=$3

CACHE_PATH=".cache/${MODEL_NAME}-${SPLIT}-${CORRUPTION}-mistral-7b-cache.db"

# First evaluate FActScore (OR SAFE)
eval_method=factscore

# If the split is expertqa, use SAFE
if [ $SPLIT == "expertqa" ]; then
    eval_method=safe
fi

export INPUT_PATH="data/new_datasets_tuned_generation/${MODEL_NAME}-${SPLIT}-${CORRUPTION}.jsonl"
export OUTPUT_PATH="data/scores/new_data/${MODEL_NAME}-${SPLIT}-${CORRUPTION}-${eval_method}.json"

# run if OUTPUT_PATH does not exist
if [ ! -f ${OUTPUT_PATH} ]; then
    conda run -p .env --no-capture-output \
        python3 scripts/run_task.py \
            configs/score/${eval_method}.yaml \
            --cache-path ${CACHE_PATH}
else
    echo "Skipping ${MODEL_NAME}-${SPLIT}-${CORRUPTION}-${eval_method}.json"
fi

# Then evaluate Core

export OUTPUT_PATH="data/scores/new_data/${MODEL_NAME}-${SPLIT}-${CORRUPTION}-core.json"

if [ ! -f ${OUTPUT_PATH} ]; then
    conda run -p .env --no-capture-output \
        python3 scripts/run_task.py \
            configs/score/core_${SPLIT}.yaml \
            --cache-path ${CACHE_PATH}
else
    echo "Skipping ${MODEL_NAME}-${SPLIT}-${CORRUPTION}-core.json"
fi