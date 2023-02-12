FROM ubuntu:20.04 as base
ARG PYTHON_VERSION=3.8

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -y ca-certificates \
         cmake \
         gcc \
         patchelf libosmesa6-dev libgl1-mesa-glx libglfw3 \
         git \
         curl \
         zip \
         unzip \
         libopenmpi-dev \
         python3-dev \
         zlib1g-dev \
         libgl1-mesa-glx \
         libtiff5-dev libjpeg8-dev libopenjp2-7-dev zlib1g-dev \
         libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python3-tk \
         libharfbuzz-dev libfribidi-dev && \
     rm -rf /var/lib/apt/lists/*

# Install Anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create --name py38 python=$PYTHON_VERSION

ENV PATH /opt/conda/bin:$PATH

RUN mkdir ~/.mujoco && \
    cd ~/.mujoco && \
    curl -o mujoco200_linux.zip https://www.roboti.us/download/mujoco200_linux.zip && \
    unzip mujoco200_linux.zip && \
    mv mujoco200_linux mujoco200 && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco200/bin" >> ~/.bashrc && \
    curl -o ./mjkey.txt https://www.roboti.us/file/mjkey.txt


RUN git clone https://github.com/XinyuCai1919/pixel-representations-RL.git && \
        conda init bash && . ~/.bashrc && \
        conda activate py38 && \
        cd  ./pixel-representations-RL && \
        pip install -r requirements.txt && \
        pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
#    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN echo "conda activate py38" >> ~/.bashrc
