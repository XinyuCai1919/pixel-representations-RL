MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=$3 python train.py \
    --domain_name cheetah --task_name run --case $1 --work_dir ./log \
    --action_repeat 4 --frame_stack 3 --data_augs $2  \
    --seed $4 --critic_lr 2e-4 --actor_lr 2e-4 --decoder_lr 2e-4 --encoder_lr 2e-4\
    --batch_size 128 --num_train_steps 200000 --metric_loss \
    --resource_files './distractors/driving/*.mp4' --img_source 'video' --total_frames 50 \
    --horizon 1 --save_model --save_tb
