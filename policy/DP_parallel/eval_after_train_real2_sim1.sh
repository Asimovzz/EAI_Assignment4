#!/bin/bash

set -u
set -o pipefail

ROOT=/root/workplace/Assignment4
TRAIN_PID=19551
CKPT_DIR="$ROOT/policy/DP_parallel/checkpoints/beat_block_hammer-real2-sim1-right8-1200-real2_sim1_ft200-beat_block_hammer-robot_dp_parallel-train-0"
LOG_DIR="$ROOT/policy/DP_parallel/eval_logs/real2_sim1_ft200"
SUMMARY_LOG="$LOG_DIR/watcher.log"

mkdir -p "$LOG_DIR"
exec >>"$SUMMARY_LOG" 2>&1

echo "[$(date '+%F %T')] watcher started for training PID $TRAIN_PID"

while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if [ -r "/proc/$TRAIN_PID/cmdline" ]; then
        cmdline=$(tr '\0' ' ' < "/proc/$TRAIN_PID/cmdline")
        case "$cmdline" in
            *train.py*real2_sim1_ft200*dataloader.batch_size=48*) ;;
            *)
                echo "[$(date '+%F %T')] PID $TRAIN_PID no longer matches the target training command"
                exit 1
                ;;
        esac
    fi
    sleep 60
done

echo "[$(date '+%F %T')] training process exited"

if [ ! -s "$CKPT_DIR/200.ckpt" ]; then
    echo "[$(date '+%F %T')] final checkpoint missing; skip evaluation: $CKPT_DIR/200.ckpt"
    exit 1
fi

export PATH=/root/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CUDA_VISIBLE_DEVICES=0
cd "$ROOT" || exit 1

failed=0
for epoch in 50 100 150 200; do
    ckpt="$CKPT_DIR/$epoch.ckpt"
    eval_log="$LOG_DIR/ckpt_${epoch}.log"

    if [ ! -s "$ckpt" ]; then
        echo "[$(date '+%F %T')] checkpoint missing: $ckpt"
        failed=1
        continue
    fi

    echo "[$(date '+%F %T')] evaluating $ckpt"
    PYTHONWARNINGS=ignore::UserWarning \
    /root/miniconda3/bin/python script/eval_policy.py \
        --config policy/DP_parallel/deploy_policy.yml \
        --overrides \
        --policy_name DP \
        --task_name beat_block_hammer \
        --task_config galbot_demo_clean \
        --ckpt_setting "real2_sim1_ft200_ckpt${epoch}" \
        --seed 0 \
        --instruction_type unseen \
        --expert_data_num 1200 \
        --checkpoint_num "$epoch" \
        --ckpt_file "$ckpt" \
        --action_dim 8 \
        --eval_video_log True \
        --eval_test_num 10 \
        --eval_step_lim 200 \
        --force_arm_tag right \
        --force_block_arm_tag right \
        2>&1 | tee "$eval_log"

    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "[$(date '+%F %T')] evaluation failed for epoch $epoch with status $status"
        failed=1
    else
        echo "[$(date '+%F %T')] evaluation completed for epoch $epoch"
    fi
done

echo "[$(date '+%F %T')] all scheduled evaluations finished; failed=$failed"
exit "$failed"
