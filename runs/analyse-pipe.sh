#!/bin/bash

set -i
set -x
set -e

# This differs from full-pipe in that it does not submit training jobs
job_id=

for model_type in mistral gpt2; do
    for split in info repetitive; do
        if [ -z $job_id ]; then
            job_op=$(sbatch runs/score_all.sh ${split} ${model_type})
            job_id=$(echo $job_op | awk '{print $4}')
        else
            job_op=$(sbatch --dependency=afterok:$job_id runs/score_all.sh ${split} ${model_type})
            job_id=$(echo $job_op | awk '{print $4}')
        fi
        job_op=$(sbatch --dependency=afterok:$job_id runs/analyse.sh ${split} ${model_type})
        job_id=$(echo $job_op | awk '{print $4}')
    done

    # run none if mistral
    if [ $model_type == "mistral" ]; then
        job_op=$(sbatch --dependency=afterok:$job_id runs/score_all.sh null ${model_type})
        job_id=$(echo $job_op | awk '{print $4}')
        job_op=$(sbatch --dependency=afterok:$job_id runs/analyse.sh null ${model_type})
        job_id=$(echo $job_op | awk '{print $4}')
    fi
done