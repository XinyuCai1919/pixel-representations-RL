MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=0 python train.py  batch_size=512 action_repeat=8 env=cartpole_swingup seed=23 &
MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=1 python train.py  batch_size=512 action_repeat=8 env=cartpole_swingup seed=57 &
MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=2 python train.py  batch_size=512 action_repeat=8 env=cartpole_swingup seed=89 &
MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=3 python train.py  batch_size=512 action_repeat=8 env=cartpole_swingup seed=111