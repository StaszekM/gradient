# Gradient

Project bootstrapped using Python `3.8.17` (default, Jun  6 2023, 20:10:50) 
[GCC 11.3.0] on linux WSL.

Virtual environment and dependency management provided by an external container.
Virtual environment and dependency management provided by an external container.

Project bootstrap command below:
Project bootstrap command below:

```console
$ git lfs install
```

Large files management provided by `git-lfs`


# Set up your Docker env

Docker image contains python3.8 set up on Ubuntu with NVIDIA, supplied with drivers to access host's CUDA if applicable. Tested only on Linux with NVIDIA GPU and WSL without NVIDIA GPU.

## Install NVIDIA drivers bridge between Docker and Linux

Follow official documentation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

## If you don't have NVIDIA CUDA on your machine...

MPS was not tested nor researched, so it is not guaranteed to work. If you intend to use CPU only, create `docker-compose.override.yml` in root folder and set its contents to:

```yaml
services:
  venv:
    runtime: runc
```

`runc` is a default non-CUDA runtime for containers and should be present in your docker distribution. All runtimes available can be seen on `docker info`.

## Running and setup after run

```bash
UNAME=$(id -un) GID=$(id -g) USERID=$(id -u) docker compose up
``` 

Root folder of the projecy is bind-mounted to `/app` inside container.

To access container file system in VS Code, install Docker extension for VS Code (https://code.visualstudio.com/docs/containers/overview), then find your container on extension's list, right click and choose `Attach to Visual Studio Code`.

Install dependencies from `/app/requirements.txt` with `python3.8 -m pip install -r /app/requirements.txt` (note the proper prefixed Python version)

## Using the container

After attaching VS Code, you can navigate in container's file system normally, run Jupyters and so on. Just make sure you're using 3.8 interpreter, not 3.10 (it is installed by default in base image, but is in initial state and has no required deps).

## Do's and Dont's, known issues

1. Do install packages with `python3.8 -m pip install`,
2. Do include installed package version in `requirements.txt`,
3. Don't include packages that are internal packages not installed by you, let pip resolve conflicting dependencies by itself
4. Python starts in container at `/` directory, so all relative imports from source code might not be found. If this is the case, for example in Jupyter, add cell with `os.chdir('/app')` to fix this.
5. Add .env file with your WANDB_API_KEY.

# Set up your Docker env

Docker image contains python3.8 set up on Ubuntu with NVIDIA, supplied with drivers to access host's CUDA if applicable. Tested only on Linux with NVIDIA GPU and WSL without NVIDIA GPU.

## Install NVIDIA drivers bridge between Docker and Linux

Follow official documentation: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html

UNAME=$(id -un) GID=$(id -g) USERID=$(id -u) docker compose up --force-recreate --build

**Please verify that your local git hooks work**: try to create a branch with random name or try to commit with a random message in terminal and check if you receive an error message and commit does not occur.

# Download organized datasets

The notebook at `data/organized-datasets/download.ipynb` can be used to recreate datasets if necessary (**attention: this process may last up to 48 hours!**).

The notebook is executed with different parameters via [Papermill](https://github.com/nteract/papermill) and results in hierarchical folder structure.

To run the notebook, open the `src/organized_datasets_creation/papermill_download.ipynb`, modify `notebook_location` and `root_project_location` variables, and run the following command:
```bash
python -m src.organized_datasets_creation.papermill_download
```

# Run project

Start by copying `config.dist.yml` to `config.yml` and all config files `downstream_task_config/[name].dist.yml` to `downstream_task_config/[name].yml`.

Fill out the config with your variables, and then run the project:

```bash
./run.sh --notebook-output [your_desired_output_ipynb] --wandb-key [wandb_key]
```

All output will be passed to stdout, and notebook with results will be created.