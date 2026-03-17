#!/bin/bash
# MuJoCo Playground 训练脚本
# 用法: bash run_mujoco_train.sh [env_name] [obs_type] [seed] [device]

ENV_NAME=${1:-CartpoleBalance}
OBS_TYPE=${2:-state}
SEED=${3:-1}
DEVICE=${4:-0}

echo "========================================="
echo " MuJoCo Playground Training"
echo " Env:      $ENV_NAME"
echo " Obs Type: $OBS_TYPE"
echo " Seed:     $SEED"
echo " Device:   $DEVICE"
echo "========================================="

export HYDRA_FULL_ERROR=1

python train.py \
    training.env_name="$ENV_NAME" \
    training.env_type=mujoco_playground \
    training.obs_type="$OBS_TYPE" \
    training.seed="$SEED" \
    training.device="$DEVICE"
