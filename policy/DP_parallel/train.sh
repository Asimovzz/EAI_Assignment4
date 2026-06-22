#!/bin/bash

set -e

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
action_dim=${5}
gpu_ids=${6}
shift 6

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_${action_dim}
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp_parallel-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"
if [ "${action_dim}" = "8" ]; then
    zarr_path="data/${task_name}-${task_config}-8d-${expert_data_num}.zarr"
else
    zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr"
fi

resume_override=""
resume_ckpt_path=""
extra_args=()
for override in "$@"; do
    case "${override}" in
        task.dataset.zarr_path=*)
            zarr_path="${override#task.dataset.zarr_path=}"
            ;;
        training.resume=*)
            resume_override="${override#training.resume=}"
            extra_args+=("${override}")
            ;;
        training.resume_ckpt_path=*)
            resume_ckpt_path="${override#training.resume_ckpt_path=}"
            extra_args+=("${override}")
            ;;
        resume_ckpt_path=*)
            resume_ckpt_path="${override#resume_ckpt_path=}"
            extra_args+=("training.resume_ckpt_path=${resume_ckpt_path}")
            ;;
        *)
            extra_args+=("${override}")
            ;;
    esac
done
if [[ "${zarr_path}" = /* ]]; then
    zarr_dir="${zarr_path}"
else
    zarr_dir="./${zarr_path}"
fi

echo -e "\033[33mgpu ids (to use): ${gpu_ids}\033[0m"

if [ "${DEBUG}" = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_ids}

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
nproc_per_node=${#gpu_array[@]}
if [ "${nproc_per_node}" -lt 1 ]; then
    echo "No GPU ids provided"
    exit 1
fi

if [ ! -d "${zarr_dir}" ]; then
    process_args=("${task_name}" "${task_config}" "${expert_data_num}" --save-dir "${zarr_path}")
    if [ "${action_dim}" = "8" ]; then
        process_args+=(--right-arm-only)
    fi
    bash process_data.sh "${process_args[@]}"
fi

if [ -n "${resume_ckpt_path}" ] && [ ! -f "${resume_ckpt_path}" ]; then
    echo "Resume checkpoint not found: ${resume_ckpt_path}"
    exit 1
fi

if [ -n "${resume_ckpt_path}" ] && [ -z "${resume_override}" ]; then
    extra_args=("training.resume=True" "${extra_args[@]}")
fi

launcher=(python train.py)
ddp_flag=False
if [ "${nproc_per_node}" -gt 1 ]; then
    launcher=(torchrun --standalone --nproc_per_node="${nproc_per_node}" train.py)
    ddp_flag=True
fi

"${launcher[@]}" --config-name=${config_name}.yaml \
    task.name=${task_name} \
    task.dataset.zarr_path="${zarr_path}" \
    training.debug=${DEBUG} \
    training.seed=${seed} \
    training.device="cuda:0" \
    training.use_ddp=${ddp_flag} \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    setting=${task_config} \
    expert_data_num=${expert_data_num} \
    head_camera_type=${head_camera_type} \
    "${extra_args[@]}"
