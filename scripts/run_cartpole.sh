MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=$3 python train.py --encoder_type $5 \
    --domain_name cartpole --task_name swingup --case $1 --work_dir ./log \
    --action_repeat 8 --frame_stack 3 --data_augs $2  \
    --seed $4 --critic_lr 1e-3 --actor_lr 1e-3 \
    --batch_size 512 --num_train_steps 200000 --metric_loss \
    --resource_files './distractors/driving/*.mp4' --img_source 'video' --total_frames 50 \
    --horizon 1 --mask_ratio $6 --save_tb
