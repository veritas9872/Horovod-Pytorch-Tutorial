# Explanation of script files

### Introduction

Three scripts are provided in this folder.

Each script must be executed in order.

Each script and its contents are explained here.

## 1. Build Image

Build the Docker image with the docker_build_script.

The -t indicates the tag/name. The format is (repository):(tag).

The full path to the Dockerfile must be specified.

Full documentation available 
[here](https://docs.docker.com/engine/reference/commandline/build/).

This might take some time.

## 2. Run Docker (Create Container)

Unfortunately for Pycharm Professional users (such as myself), 
I could not find a way to run Horovod properly with the Pycharm GUI.

The docker_run_script creates a container that can execute files on the local device.

See [here](https://docs.docker.com/engine/reference/run) 
for an explanation of the docker run command.

##### Flags
- The -v flag creates a [volume](https://docs.docker.com/storage/volumes) 
that allows the local project files to be accessed by the container and vice versa.
The structure is (host path):(container path). 
The container path is set to "/opt/project", the default location for Pycharm.

- The -it flag allows command-line interaction.

- The -w flag changes working directory in the container. 
This is set to "/opt/project", the path containing all project files.

- The --rm flag removes the container after exiting.
Be aware that volume data (saved logs, checkpoints, etc.) will remain even if containers are removed.

- The --name flag sets the name of the container. 
If omitted, Docker will generate a name for you. 

#### Using GPUs on the container

NVIDIA-Docker adds several extra flags and options to the defaults.

The container runtime [documentation](https://github.com/NVIDIA/nvidia-container-runtime/blob/v3.1.4/README.md) 
contains detailed explanations of options.

To allow GPUs to be used, either specify `--runtime=nvidia` or set the `--gpus` flag to some value.

The `--runtime` flag is unnecessary if the `--gpus` flag is specified.

Setting only `--runtime=nvidia` is equivalent to `--gpus=all`.

For exact usage, see the official NVIDIA-Docker 
[documentation](https://github.com/NVIDIA/nvidia-docker/blob/3f1edae37ea46c030b5585bca4ce524da51c06c7/README.md).

Options such as setting the NVIDIA runtime as the default or using only specific devices are also available.

## 3. Run Horovod

For an in-depth guide please visit the official 
[documentation](https://horovod.readthedocs.io/en/latest/).

By the time you read this, it may have improved considerably.

Major features are the `-np` and `-H` flags.

**1. Number of Processes.**

The `-np` flag stands for "number of processes".

This value should be set to be the same as the number of GPUs.

If it is set to below the number of GPUs, some GPUs will be idle.

If more than the number of GPUs, the same GPU will have multiple processes.

This will cause an error because there are more processes than slots (GPUs) available.

**Important!**

`-np` dictates the number of processes that __*Horovod*__ launches, 
not the number that __*Python*__ launches.

In Pytorch, the DataLoader class uses multi-processing for efficient data pre-processing.

This is specified by the `num_workers` variable.

However, each Horovod process will launch these workers independently.

This may cause an excessive number of workers to be launched. 

**2. Host**

The `-H` flag specifies the host type. 

The number of GPUs to be used is specified on the right.

N must be the same or lesser than the number of GPUs.

For a local machine where N GPUs are to be used, use "localhost:N".

GPUs 0~N-1 will be utilized in that run.

For servers, a different scheme is used.

For server with index I with N GPUs, use "serverI:N".

For large servers, use a hostfile.

**3. Autotune**

Use the `--autotune` flag to autotune parameters for best performance.

Autotuning uses Bayesian optimization for finding the best parameters.

This will cause early runs to be slower, but later runs will be faster.


---
For up-to-date and detailed explanations please visit the official 
[documentation](https://horovod.readthedocs.io/en/latest/).

__*Please star/fork my repository if you find my explanations helpful!*__