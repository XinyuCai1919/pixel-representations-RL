# Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models (NeurIPS 2022 Spotlight)
A PyTorch implementation of our paper: 

#### Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models.

Minting Pan*, Xiangming Zhu*, Yunbo Wang, Xiaokang Yang

[[arXiv]](https://arxiv.org/abs/2205.13817)  [[Project Page]](https://sites.google.com/view/iso-dream)

## Showcases

#### DMC and CARLA

<img src="https://github.com/panmt/Iso-Dream/blob/main/picture/dmc_carla_vis.png" width="70%" align=“center” />

Demo of Iso-Dream in DMC with noisy video backgrounds

<img src="https://github.com/panmt/Iso-Dream/blob/main/picture/dmc_demo.gif" width="30%" align=“center” /> <img src="https://github.com/panmt/Iso-Dream/blob/main/picture/dmc_demo_2.gif" width="30%" align=“center” />

Demo of Iso-Dream in CARLA

<img src="https://github.com/panmt/Iso-Dream/blob/main/picture/carla_demo.gif" width="30%" align=“center” /> <img src="https://github.com/panmt/Iso-Dream/blob/main/picture/carla_demo_2.gif" width="30%" align=“center” />


#### BAIR

<img src="https://github.com/panmt/Iso-Dream/blob/main/picture/bair_vis.png" width="85%" align=“center” />

## Get Started
Iso-Dream is implemented and tested on Ubuntu 18.04 with python == 3.7, PyTorch == 1.9.0:

1. Create an environment 
   ```
   conda create -n iso-env python=3.7
   conda activate iso-env
   ```   

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

### DMC / CARLA

#### For CARLA environment:

  1. Setup
  
     Download and setup CARLA 0.9.10
     ```
     chmod +x setup_carla.sh
     ./setup_carla.sh
     ```
     
     Add to your python path:
     ```
     export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI
     export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI/carla
     export PYTHONPATH=$PYTHONPATH:/home/CARLA_0.9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
     ```
     and merge the directories, i.e., put 'carla_env_dream.py' into 'CARLA_0.9.10/PythonAPI/carla/agents/navigation/'.

  2. Training
  
     Terminal 1:
     ```
     cd CARLA_0.9.10
     bash CarlaUE4.sh -fps 20 -opengl
     ```

     Terminal 2:
     ```
     cd dmc_carla_iso
     python dreamer.py --logdir log/iso_carla --action_step 20 --free_step 50 --kl_balance 0.8 --action_scale 1 --seed 9 --configs defaults carla
     ```

  3. Evaluation
     ```
     cd dmc_carla_iso
     python test.py --logdir test --action_step 20 --free_step 50 --kl_balance 0.8 --configs defaults carla
     ```

#### For DMC environment:

  1. Setup DMC with video background
  
     Download 'envs' from [Google Drive](https://drive.google.com/drive/folders/1vAHRBx7zlK-XHowSOAv-gBPWlubvpnCo?usp=sharing) and put it in the 'dmc_carla_iso'. The dependencies can then be installed with the following commands:
  
     ```
     cd dmc_carla_iso
     
     cd ./envs/dm_control
     pip install -e .
     
     cd ../dmc2gym
     pip install -e .

     cd ../..
     ```

  2. Training
     ```
     python dreamer.py --logdir log/iso_dmc --kl_balance 0.8 --seed 4 --configs defaults dmc --task dmcbg_walker_walk
     ```
  

### BAIR / RoboNet
Train and test Iso-Dream on BAIR and RoboNet datasets. Also, install Tensorflow 2.1.0 for BAIR dataloader.

1. Download BAIR data. 
   ```
   wget http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
   ```

2. Train the model. You can use the following bash script to train the model. The learned model will be saved in the `--save_dir` folder.
  The generated future frames will be saved in the `--gen_frm_dir` folder. 
    ```
    cd bair_robonet_iso
    sh train_iso_model.sh
    ```

## Citation

Pan, Minting and Zhu, Xiangming and Wang, Yunbo and Yang, Xiaokang. "Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models". Advances in Neural Information Processing Systems. 2022.

```
@inproceedings{paniso,
  title={Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models},
  author={Pan, Minting and Zhu, Xiangming and Wang, Yunbo and Yang, Xiaokang},
  booktitle={Advances in Neural Information Processing Systems}
  year={2022}
}
```

## Acknowledgement
We appreciate the following github repos where we borrow code from:

https://github.com/jsikyoon/dreamer-torch

https://github.com/thuml/predrnn-pytorch



