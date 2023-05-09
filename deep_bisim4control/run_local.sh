#!/bin/bash

DOMAIN=$1
TASK=$2
REPEAT=$3
SEED=$4
CUDA=$5

SAVEDIR=./save

MUJOCO_GL="osmesa"  CUDA_VISIBLE_DEVICES=${CUDA} python train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --action_repeat ${REPEAT} \
    --save_tb \
    --save_model \
    --work_dir ${SAVEDIR}/${DOMAIN}_${TASK}_${SEED} \
    --seed ${SEED} &