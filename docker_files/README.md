# Explanation of Dockerfile

The Dockerfile here is used to create an environment where Horovod can be used with Pytorch.

Software versions have been fixed for my convenience. 

However, they can be changed manually.

The current installation uses pip instead of Anaconda.

This is keeping with the original Dockerfile in Horovod.

See [here](https://github.com/horovod/horovod/blob/f7e8d4e007329508a3d3d4f82c24e10487b6b27a/Dockerfile.gpu)
for the original.

### Dependencies and installation

The section before `pip install ...` is boilerplate for dependencies on Ubuntu.

The current project only uses Pytorch, Torchvision, Tensorboard, Numpy, and typing.

The "future" library is necessary when installing Tensorboard with pip.

Pytorch and Torchvision are installed with their wheel directories in PyPI for future-proofing.

See [here](https://download.pytorch.org/whl/cu100/torch_stable.html) for Pytorch wheels for CUDA10.0.

Other project requirements should be installed here.

### Horovod installation

Horovod is installed using three flags. 

These are: 
HOROVOD_GPU_ALLREDUCE=NCCL, 
HOROVOD_GPU_BROADCAST=NCCL, and 
HOROVOD_WITH_PYTORCH=1.

The first two indicate that GPU operations should use NCCL (pronounced "nickel").

This setting is crucial for performance.

NCCL is NVIDIA's highly optimized library for multi-GPU operations. 

The last flag indicates that Pytorch must be installed.

### Build

To build the Docker image, use the docker_build_script in scripts.

Build process will take maybe 20 minutes.

__*Please star/fork my repository if you find this tutorial helpful!*__