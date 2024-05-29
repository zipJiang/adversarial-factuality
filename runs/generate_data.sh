#!/bin/bash
#SBATCH --job-name=data-generation
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --partition=brtx6

export PARENT_DIR=/brtx/606-nvme2/zpjiang/adversarial-factuality/
export PYTHONPATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/:$PYTHONPATH
export ENV_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/.env
INPUT_STEM=/brtx/606-nvme2/zpjiang/adversarial-factuality/data/entity_filtering/

export SPLIT=$1

"/brtx/606-nvme2/zpjiang/adversarial-factuality/runs/port_binding.sh"

for filename in sampled_dev.txt sampled_train.txt sampled_test.txt; do
    echo "Processing $filename"
    export INPUT_PATH=$INPUT_STEM$filename
    export OUTPUT_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/data/generated-samples/${filename%.txt}-${SPLIT}.jsonl
    conda run -p $ENV_PATH --no-capture-output \
        python3 ${PARENT_DIR}scripts/run_task.py ${PARENT_DIR}configs/generation/sample_${SPLIT}_violation.yaml

    # export OUTPUT_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/data/generated-samples/${filename%.txt}-relevance.jsonl
    # conda run -p $ENV_PATH --no-capture-output \
    #     python3 scripts/run_task.py configs/generation/sample_relevancy_violation.yaml
done