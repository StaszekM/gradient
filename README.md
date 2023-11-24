# Gradient

Project bootstrapped using Python `3.8.17` (default, Jun  6 2023, 20:10:50) 
[GCC 11.3.0] on linux WSL.

Virtual environment and dependency management provided by an external container.

Project bootstrap command below:

```console
$ git lfs install
```

Large files management provided by `git-lfs`


# Set up your Docker env

Docker image contains python3.8 set up on Ubuntu with NVIDIA, supplied with drivers to access host's CUDA if applicable. Tested only on Linux with NVIDIA GPU and WSL without NVIDIA GPU.

## Install NVIDIA drivers bridge between Docker and Linux

Follow official documentation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

UNAME=$(id -un) GID=$(id -g) USERID=$(id -u) docker compose up --force-recreate --build
