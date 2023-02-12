FROM ubuntu:20.04 as base
ARG PYTHON_VERSION=3.6

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends --no-install-suggests -y ca-certificates \
         cmake \
         git \
         curl \
         zip \
         unzip \
         libopenmpi-dev \
         python3-dev \
         zlib1g-dev \
         libgl1-mesa-glx \
         swig && \
     rm -rf /var/lib/apt/lists/*

# Install Anaconda and dependencies
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda create --name py36 python=$PYTHON_VERSION

ENV PATH /opt/conda/bin:$PATH

#RUN mkdir ~/.mujoco && \
#    cd ~/.mujoco \
#    curl -o ./mujoco200.zip https://www.roboti.us/download/mujoco200_linux.zip && \
#    unzip mujoco200.zip && \
#    curl -o ./mjkey.txt mjkey.txt && \
#
#
#RUN git clone https://github.com/XinyuCai1919/pixel-representations-RL.git && \
#        cd  ./pixel-representations-RL && \
#        conda init bash && . ~/.bashrc && \
#        conda activate py36 && \
#        pip install -r requirements.txt && \
#        pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
##    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

RUN echo "conda activate py36" >> ~/.bashrc
