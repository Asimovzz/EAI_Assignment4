#!/bin/bash

set -e

if [ "$#" -lt 6 ]; then
    echo "Usage: bash train_ddp.sh <task_name> <task_config> <expert_data_num> <seed> <action_dim> <gpu_ids> [hydra overrides...]"
    echo "Example: bash train_ddp.sh beat_block_hammer default 100 42 8 0,1,2,3 dataloader.batch_size=32"
    exit 1
fi

bash train.sh "$@"
