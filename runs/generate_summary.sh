#!/bin/bash

export PYTHONPATH=$(pwd):$PYTHONPATH
export PROMPT="Tell me a bio of {input}."
export OUTPUT_PATH=data/outputs/generation_opt.json
# export PROMPT="Tell me a bio of {input}. If you are very certain about a fact, try to putting more emphasis on it by repeating that fact in multiple sentences."
# export OUTPUT_PATH=data/outputs/generation_opt_repeat.json
# export PROMPT="Tell me something tautological, obvisouly true and easily verifiable about {input}. Repeat that fact multiple times in paraphrased sentences."
# export OUTPUT_PATH=data/outputs/generation_opt_trivial.json

# This does not need to be replaced
SCORE_PATH=${OUTPUT_PATH/opt/scoring}
export SCORE_PATH=${SCORE_PATH/outputs/scores}

conda run -p .env --no-capture-output \
    python3 scripts/run_task.py configs/generation_configs.yaml

# conda run -p .env --no-capture-output \
#     python3 scripts/run_task.py configs/scoring_configs.yaml