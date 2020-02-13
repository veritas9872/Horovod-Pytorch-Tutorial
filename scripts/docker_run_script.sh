set -ex
docker run -v $HOME/PycharmProjects/Horovod-Pytorch-Tutorial:/opt/project \
            -it -w /opt/project --runtime nvidia --gpus all --name horovod_torch --rm \
            horovod:py-3.6-torch-1.4.0