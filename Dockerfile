# Use official NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install Python, pip, and git
RUN apt update && apt install -y python3 python3-pip git

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Set default command to run training
CMD ["python3", "main.py"] 