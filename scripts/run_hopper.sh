MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=$2 python train.py \
    --domain_name hopper --task_name hop --case $1 --work_dir ./log \
    --action_repeat 4 --frame_stack 3 --data_augs no_aug  \
    --seed $3 --critic_lr 1e-3 --actor_lr 1e-3 \
    --batch_size 128 --num_train_steps 200000 --metric_loss \
    --resource_files './distractors/driving/*.mp4' --img_source 'video' --total_frames 50 \
    --horizon $2 --save_model --save_tb
