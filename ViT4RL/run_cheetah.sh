MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=0 python train.py  batch_size=512 lr=1e-3 action_repeat=4 env=cheetah_run seed=23 &
MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=1 python train.py  batch_size=512 lr=1e-3 action_repeat=4 env=cheetah_run seed=57 &
MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=2 python train.py  batch_size=512 lr=5e-4 action_repeat=4 env=cheetah_run seed=89 &
MUJOCO_GL="osmesa" CUDA_VISIBLE_DEVICES=3 python train.py  batch_size=512 lr=5e-4 action_repeat=4 env=cheetah_run seed=111

