#!/bin/bash

set -e

if [ "$#" -lt 7 ]; then
    echo "Usage: bash train_resume.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_ids> <resume_ckpt_path> [hydra overrides...]"
    echo "Full resume example:"
    echo "  bash train_resume.sh beat_block_hammer real_keep100 100 0 8 0,1,2,3 /path/to/50.ckpt dataloader.batch_size=12"
    echo "Weights-only finetune example:"
    echo "  bash train_resume.sh beat_block_hammer mixed 400 0 8 0,1,2,3 /path/to/200.ckpt training.resume_optimizer=False training.resume_training_state=False training.num_epochs=50 optimizer.lr=2e-5"
    exit 1
fi

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_ids=${6}
resume_ckpt_path=${7}
shift 7

bash train.sh \
    "${task_name}" \
    "${task_config}" \
    "${expert_data_num}" \
    "${seed}" \
    "${action_dim}" \
    "${gpu_ids}" \
    "training.resume=True" \
    "training.resume_ckpt_path=${resume_ckpt_path}" \
    "$@"
