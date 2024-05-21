#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export PROMPT="Tell me a bio of {input}."

export TOPIC_PATH=/weka/scratch/jzhan237/repos/adversarial-factuality/inputs/bio1024/train_401to800_dup5.txt
export OUTPUT_PATH=outputs/gens/bio1024/train_401to800_dup5_mistral-inst_train1to400subset_beta005.json
# export PROMPT="Tell me a bio of {input}. If you are very certain about a fact, try to putting more emphasis on it by repeating that fact in multiple sentences."
# export OUTPUT_PATH=data/outputs/generation_opt_repeat.json
# export PROMPT="Tell me something tautological, obviously true and easily verifiable about {input}. Repeat that fact multiple times in paraphrased sentences."
# export OUTPUT_PATH=data/outputs/generation_opt_trivial.json

# This does not need to be replaced
SCORE_PATH=${OUTPUT_PATH/opt/scoring}
export SCORE_PATH=${SCORE_PATH/outputs/scores}

# conda run -p .env --no-capture-output \
conda run -n advfact --no-capture-output \
    python3 scripts/run_task.py configs/generation_configs.yaml

# conda run -p .env --no-capture-output \
#     python3 scripts/run_task.py configs/scoring_configs.yaml