FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx

# Clone the YOLOX repository
COPY /YOLOX/requirements.txt /app/requirements.txt

WORKDIR /app

# Install Python dependencies
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter

COPY /YOLOX /app/YOLOX

WORKDIR /app/YOLOX

# Install YOLOX in development mode
RUN pip3 install -v -e .

WORKDIR /app
RUN rm -rf YOLOX

# Set the default command to run a terminal
CMD ["bash"]