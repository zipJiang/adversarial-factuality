#!/bin/bash
#SBATCH --job-name=analyse-scoring-result
#SBATCH --mail-user=zjiang31@jh.edu
#SBATCH --partition=brtx6

export PARENT_DIR=/brtx/606-nvme2/zpjiang/adversarial-factuality/

export PYTHONPATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/:$PYTHONPATH
export ENV_PATH=/brtx/606-nvme2/zpjiang/adversarial-factuality/.env

export SPLIT=$1
export MODEL_TYPE=$2

conda run -p ${ENV_PATH} --no-capture-output \
    python ${PARENT_DIR}scripts/run_task.py ${PARENT_DIR}configs/misc/analyse_result.yaml