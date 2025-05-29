FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list

# Install system dependencies
RUN apt-get update
RUN apt-get install -y git curl wget ca-certificates build-essential

# Install Python packages
RUN pip install --no-cache-dir \
    openmim==0.3.9 \
    mlflow==2.17.2 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    opencv-python==4.11.0.86 \
    pycocotools==2.0.7 \
    lxml \
    pandas==2.0.3 \
    matplotlib==3.7.5 \
    scipy==1.10.1 \
    scikit-learn==1.3.2 \
    tqdm==4.65.2 \
    requests==2.28.2

# Install MMDetection stack
RUN mim install mmengine mmcv-full mmdet