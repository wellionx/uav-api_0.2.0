################## BASE IMAGE ######################
FROM nvidia/cuda:12.2.0-base-ubuntu20.04

################## METADATA ######################
LABEL base_image="nvidia/cuda:12.2.0-base-ubuntu20.04"
LABEL version="1"
LABEL software="UAV Image Stitching and Authentication API"
LABEL about.summary="Environment for UAV Image Stitching API with authentication"
LABEL about.license="BioBin Private"
LABEL about.tags="UAV-API"

################## MAINTAINER ######################
MAINTAINER Yirong Yang <yirong.yang@biobin.com.cn>

ENV DEBIAN_FRONTEND=noninteractive

# Copy Miniconda installer
COPY Miniconda3-py39_24.3.0-0-Linux-x86_64.sh /tmp/miniconda.sh

# Install system dependencies and Miniconda
RUN apt update && \
    apt install -y libx11-dev libxrender-dev libxt-dev libxext-dev libcairo2-dev && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh

# Copy the Metashape wheel file
COPY Metashape-2.1.2-cp39-cp39-linux_x86_64.whl /tmp/

# Install Python dependencies
RUN pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install /tmp/Metashape-2.1.2-cp39-cp39-linux_x86_64.whl

# Copy application code
COPY . /app
WORKDIR /app

# Expose the port the app runs on
EXPOSE 8081

# Run the application
CMD ["gunicorn", "-w", "1", "--threads", "64", "-b", "0.0.0.0:5000", "--access-logfile", "./log/access.log", "--error-logfile", "./log/error.log", "-t", "3600", "app:app"]