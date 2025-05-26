ARG PYTORCH="2.0.1"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

# Avoid Public GPG key error
# https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    && rm -f /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 || true \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

# Install the required packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMEngine and MMCV
RUN pip install openmim && \
    mim install "mmengine>=0.7.1" "mmcv==2.0.0"

# Set up working directory
WORKDIR /mmdetection

# Copy your existing mmdetection directory
COPY . /mmdetection/

# Install mmdetection in development mode
RUN pip install --no-cache-dir -e .

# Install any additional requirements
RUN pip install --no-cache-dir \
    pycocotools \
    terminaltables \
    matplotlib \
    pandas \
    scipy \
    tqdm \
    tensorboard \
    mlflow

# Create output directories if they don't exist
RUN mkdir -p /mmdetection/outputs
RUN mkdir -p /mmdetection/work_dirs

# Create entrypoint script
COPY entrypoint.sh /mmdetection/
RUN chmod +x /mmdetection/entrypoint.sh

# Set default entrypoint
ENTRYPOINT ["/mmdetection/entrypoint.sh"]
CMD ["python", "/mmdetection/tools/train.py", "--help"]
