docker run -v /home/xycai/pixel-representations-RL:/opt/project --runtime=nvidia \
            -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all \
            -w /opt/project -it --rm cxy/py38:0.2  bash

docker run -v /home/xycai/pixel-representations-RL:/opt/project --gpus=all --cpus=40\
            -w /opt/project -it --rm cxy/py38:0.2  bash

