#!/bin/bash

# submit a series of jobs and manage their interdependencies
# to run a full pipeline

SPLIT=$1

# 1. generate dataset and get job ids
job_output=$(sbatch runs/generate_data.sh ${SPLIT})
job_id=$(echo $job_output | awk '{print $4}')

# 2. When the dataset generation is done, train mistral model (and gpt-2)

job_output_mistral_train=$(sbatch --dependency=afterok:$job_id runs/train_peft.sh ${SPLIT} mistral)
job_id_mistral_train=$(echo $job_output_mistral_train | awk '{print $4}')
job_output_gpt2_train=$(sbatch --dependency=afterok:$job_id runs/train_peft.sh ${SPLIT} gpt2)
job_id_gpt2_train=$(echo $job_output_gpt2_train | awk '{print $4}')

# 3. while each job finishes, these job also generates required tuned_generation, score them
# notice that to avoid database locked error, we need to wait for the previous job to finish then
# score the second.

score_job_gpt2=$(sbatch --dependency=afterok:$job_id_gpt2_train runs/score_all.sh ${SPLIT} gpt2)
score_job_gpt2_id=$(echo $score_job_gpt2 | awk '{print $4}')

# We do this later because maistral typically trains longer
score_job_mistral=$(sbatch --dependency=afterok:$job_id_mistral_train:$score_job_gpt2_id runs/score_all.sh ${SPLIT} mistral)
score_job_mistral_id=$(echo $score_job_mistral | awk '{print $4}')


# 4. After the scoring, run the analysis script

declare -A model_type_job_id
model_type_job_id["mistral"]=$score_job_mistral_id
model_type_job_id["gpt2"]=$score_job_gpt2_id

for model_type in mistral gpt2; do
    sbatch --dependency=afterok:${model_type_job_id[$model_type]} runs/analyse.sh ${SPLIT} ${model_type}
done