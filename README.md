# Unofficial repository for training consistency models

You can train and generate samples by running the main.py file.
For hyperparameter adjustments, just edit parameters.yaml file.

# For running on any number of nodes without docker:

**set nproc_per_node as [number of nodes] and nnodes as [desired number of GPUs].**
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```


# For running on any number of nodes with docker:

Please follow the instructions to install NVIDIA Container Toolkit before creating the docker image: 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

**set nproc_per_node as 1 and nnodes as [desired number of GPUs].**

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
sudo docker build -t consistency_docker .
sudo docker run --rm --runtime=nvidia --gpus all consistency_docker

```