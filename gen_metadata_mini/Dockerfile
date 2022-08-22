FROM ubuntu:20.04

# Label
LABEL org.opencontainers.image.title="metadata generation for fish image"
LABEL org.opencontainers.image.authors=" J. Pepper, K. Karmani, T. Tabarin"
LABEL org.opencontainers.image.source="https://github.com/hdr-bgnn/drexel_metadata"

ARG DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    sudo \
    cmake \
    ninja-build \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /pipeline
WORKDIR /pipeline

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /pipeline
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda clean -ya
 
RUN pip install numpy pandas pynrrd pillow scikit-image jedi==0.17.2 opencv-python-headless
RUN pip install --upgrade --no-cache-dir gdown

# Detectron2 prerequisites
RUN pip install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
# Install detectron2 version 0.6
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.8/index.html

# Setup pipeline specific scripts
ENV PATH="/pipeline:${PATH}"

ADD scripts/ /pipeline/
# Download Drexel Model
RUN mkdir /pipeline/output/
RUN gdown -O /pipeline/output/model_final.pth https://drive.google.com/uc?id=13pa5E5odN_gWNZYkA12u8ZEnEjzWGxFL

# Set the default command to a usage statement
CMD echo "Usage Metadata: metadata_main.py  <fish_image.jpg> <metadata.json> <mask.png>"


 
 
