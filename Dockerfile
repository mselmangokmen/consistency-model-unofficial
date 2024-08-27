# Create a Dockerfile using an Ubuntu 22.04 base image
FROM ubuntu:22.04
 
# Set the working directory to /app
WORKDIR /app

# Update the Docker image and install required packages
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt
 
COPY . /app
#RUN chmod -R 777 /app

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

    
RUN mkdir /app/outputs && \
    chmod +w /app/outputs



RUN mkdir /app/dataset && \
    chmod +w /app/dataset
