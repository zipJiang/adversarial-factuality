#!/bin/bash
#SBATCH --job-name=data-generation
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --partition=brtx6-dev

export PYTHONPATH=$(pwd):$PYTHONPATH
export ENV_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/.env
INPUT_STEM=/brtx/606-nvme2/zpjiang/adversarial-factuality/data/entity_filtering/

for split in manner; do
    for filename in sampled_dev.txt sampled_train.txt sampled_test.txt; do
        echo "Processing $filename"
        export INPUT_PATH=$INPUT_STEM$filename
        export OUTPUT_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/data/generated-samples/${filename%.txt}-${split}.jsonl
        conda run -p $ENV_PATH --no-capture-output \
            python3 scripts/run_task.py configs/generation/sample_${split}_violation.yaml

        # export OUTPUT_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/data/generated-samples/${filename%.txt}-relevance.jsonl
        # conda run -p $ENV_PATH --no-capture-output \
        #     python3 scripts/run_task.py configs/generation/sample_relevancy_violation.yaml
    done
done