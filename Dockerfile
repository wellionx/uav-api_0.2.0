# BASE DOCKER IMAGE
FROM nvidia/cuda:12.2.0-base-ubuntu20.04
USER root

# MAINTAINER
MAINTAINER <LinLin Gu, linlin.gu@biobin.com.cn, Genome Data Scientist, BDAI, BioBin>

# METADATA
LABEL base_image="nvidia/cuda:12.2.0-base-ubuntu20.04"
LABEL version="1"
LABEL software="Metashape, flask, python, R"
LABEL about.summary="Environment for UAV Image Stitching API with authentication"
LABEL about.home=""
LABEL about.documentation=""
LABEL about.license_file="Biobin Private"
LABEL about.license="BioBin Private"
LABEL about.tags="UAV-API"

# Create user account with password-less sudo abilities
RUN useradd -s /bin/bash -g 100 -G sudo -m user
RUN /usr/bin/printf '%s\n%s\n' 'password' 'password'| passwd user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV TZ=Asia/Shanghai
ENV DEBIAN_FRONTEND=noninteractive

# Copy requirements to container directory /tmp
COPY Miniconda3-py39_24.3.0-0-Linux-x86_64.sh /tmp/miniconda.sh

# Install libraries/dependencies
RUN apt-get update && \
    apt-get install -y \
      tzdata \
      libgl1-mesa-glx \
      libglu1 \
      libglu1-mesa \
      software-properties-common \
      libatk-adaptor \
      libcairo2-dev \
      libcanberra-dev libcanberra-gtk-module libcanberra-gtk3-module \
      libgl1-mesa-glx libglu1 \     
      libjpeg-turbo8 libjpeg-turbo8-dev \
      libpng-dev \
      libssl-dev \ 
      libpoppler-dev \               
      gcc \
      mesa-utils \
      gtk2.0 \
      make \
      libx11-dev \
      libxrender-dev \
      libxt-dev \
      libxext-dev \
      pixmap \
      libcurl4 \
      curl \
      libxi6 \
      libsm6 \
      libfontconfig \
      libxrender1 \
      libqt5x11extras5 \
      wget && \
      apt-get install -y --reinstall overlay-scrollbar-gtk2 && \
      rm -rf /var/lib/apt/lists/*
     
# Install the command line python module. Note that this does not install the GUI
RUN apt-get update && \
    apt-get install -y python3-pip && \
    bash /tmp/miniconda.sh -bfp /usr/local && \
    rm -rf /tmp/miniconda.sh && \
    conda install -y -c nvidia cuda-nvcc=12.4.131 && \
    conda install -y cudatoolkit=11.8.0 cudnn=8.9.2.26 && \
    pip install flask==2.2.5 pysam flask-restful==0.3.10 PyJWT==2.8.0 apispec==6.3.1 flask_apispec==0.11.4 PyYAML==6.0.1 seaborn gunicorn greenlet eventlet gevent -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    conda install r-base=4.3.1 -y && \
    conda install -y r-sf r-ggplot2 r-ggspatial r-codetools r-terra r-dplyr r-svglite && \
    pip install scikit-learn==1.3.0 pandas==1.5.2 opencv-python==4.8.1.78 matplotlib==3.7.4 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install protobuf scikit-image easyidp -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install h5py rasterio geopandas flask_jwt_extended -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    conda install -y -c conda-forge font-ttf-dejavu-sans-mono

RUN pip install numpy==1.23.5 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN ln -sf /usr/local/lib/libpoppler.so.141.0.0 /usr/local/lib/libpoppler.so.126 && \
    ldconfig
    
# Install Metashape
RUN cd /opt && wget https://s3-eu-west-1.amazonaws.com/download.agisoft.com/metashape-pro_2_1_1_amd64.tar.gz && \
     tar xvzf metashape-pro_2_1_1_amd64.tar.gz && \
     export PATH=$PATH:/opt/metashape-pro && \
       rm -rf *.tar.gz

# Install Python3 Module
RUN cd /opt && wget https://s3-eu-west-1.amazonaws.com/download.agisoft.com/Metashape-2.1.1-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl && \
      pip3 install Metashape-2.1.1-cp37.cp38.cp39.cp310.cp311-abi3-linux_x86_64.whl && pip3 install PyYAML && \
      rm -rf *.whl

CMD chmod -R 755 /opt/

ENV PATH=$PATH:/opt/metashape-pro

CMD /opt/metashape-pro/metashape.sh

# Add licenses - do not save licenses in repository
# COPY server.lic /opt/metashape-pro/server.lic

# Set the default command and default arguments
WORKDIR /uav_api

# Expose 8081 ports for server
CMD ["gunicorn", "-w", "1", "--threads", "64", "-b", "0.0.0.0:8081", "--access-logfile", "./log/access.log", "--error-logfile", "./log/error.log", "-t", "3600", "app:app"]
# gunicorn -w 1 --threads 64 -b 0.0.0.0:8081 --access-logfile ./log/access.log --error-logfile ./log/error.log -t 3600 app:app