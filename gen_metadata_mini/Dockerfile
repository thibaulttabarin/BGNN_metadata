FROM python:3.8-slim-buster

# Label
LABEL org.opencontainers.image.title="Generate metadata"
LABEL org.opencontainers.image.authors="Joel Pepper, Kevin Karmani T. Tabarin"
LABEL org.opencontainers.image.source="https://github.com/hdr-bgnn/drexel_metadata"

RUN apt-get update -y

# gcc compiler and opencv prerequisites
RUN apt-get -y install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev sudo cmake ninja-build

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Detectron2 prerequisites
RUN pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install cython
RUN pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

RUN python -m pip install detectron2==0.4 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.6/index.html

RUN pip install -U 'git+https://github.com/facebookresearch/fvcore'

RUN pip install numpy pandas pynrrd pillow scikit-image jedi==0.17.2 opencv-python-headless

WORKDIR /pipeline

# Setup pipeline specific scripts
ENV PATH="/pipeline:${PATH}"

ADD scripts/ /pipeline/

# Set the default command to a usage statement
CMD echo "Usage Metadata: python metadata_main.py  <fish_image.jpg> <metadata.json> <mask.png>"

