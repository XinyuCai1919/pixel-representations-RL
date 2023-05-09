#!/bin/bash

DOMAIN=$1
TASK=$2
SEED=$3
CUDA=$4

SAVEDIR=./save

MUJOCO_GL="osmesa"  CUDA_VISIBLE_DEVICES=${CUDA} python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --action_repeat 2 \
    --batch_size 128 \
    --save_tb \
    --save_model \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${SEED} \
    --seed ${SEED} &

. run_local.sh cartpole swingup 23 0