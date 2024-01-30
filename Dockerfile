# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.3.1-base-ubuntu20.04 as base

# Install required system packages
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libjpeg-dev \
    libpng-dev

# Copy the requirements file into the container
COPY requirements.txt /workspace/requirements.txt

# Install Python packages
RUN pip3 install --no-cache-dir -r /workspace/requirements.txt

# Set the working directory
WORKDIR /workspace

# Copy local project files to /workspace in the image
COPY . /workspace

# Expose port 6006 for TensorBoard
EXPOSE 6006
