import torch
from torch import nn, Tensor
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet34
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop

import horovod.torch as hvd


def main():
    model: nn.Module = resnet34(num_classes=10).cuda()

    # print('Number of threads: ', torch.get_num_threads(), torch.get_num_interop_threads())

    batch_size = 1024
    num_workers_per_process = 2  # Workers launched by each process started by horovodrun command.
    lr = 0.1
    momentum = 0.9
    weight_decay = 1E-4
    root_rank = 0
    num_epochs = 10

    train_transform = Compose([RandomHorizontalFlip(), RandomCrop(size=32, padding=4), ToTensor()])
    train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

    # Distributed samplers are necessary for accurately dividing the dataset among the processes.
    # It also controls mini-batch size effectively between processes.
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_sampler = DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    # Create iterable to allow manual unwinding of the for loop.
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              sampler=train_sampler, num_workers=num_workers_per_process, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                             sampler=test_sampler, num_workers=num_workers_per_process, pin_memory=True)

    loss_func = nn.CrossEntropyLoss()

    # print('Thread number: ', torch.get_num_threads(), torch.get_num_interop_threads())

    # Writing separate log files for each process. Verified that models are different.
    writer = SummaryWriter(log_dir=f'./logs/{hvd.local_rank()}', comment='Summary writer for run.')

    # Optimizer must be distributed for the Ring-AllReduce.
    optimizer = SGD(params=model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    def warm_up(epoch: int):  # Learning rate warm-up.
        if epoch < 5:
            return (epoch + 1) / 5
        elif epoch < 75:
            return 1
        elif epoch < 90:
            return 0.1
        else:
            return 0.01

    scheduler = LambdaLR(optimizer, lr_lambda=warm_up)  # Learning rate scheduling with warm-up.

    # Broadcast the model's parameters to all devices.
    hvd.broadcast_parameters(model.state_dict(), root_rank=root_rank)

    for epoch in range(num_epochs):
        print(epoch)
        torch.autograd.set_grad_enabled = True
        train_sampler.set_epoch(epoch)  # Set epoch to sampler for proper shuffling of training set.
        for inputs, targets in train_loader:
            inputs: Tensor = inputs.cuda(non_blocking=True)
            targets: Tensor = targets.cuda(non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer.step()

        torch.autograd.set_grad_enabled = False
        for step, (inputs, targets) in enumerate(test_loader):
            inputs: Tensor = inputs.cuda(non_blocking=True)
            targets: Tensor = targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            writer.add_scalar(tag='val_loss', scalar_value=loss.item(), global_step=step)

        scheduler.step()


if __name__ == '__main__':
    # torch.set_num_threads(1)  # Set threads to 1. However, threads are 1 anyway, so this is unnecessary.
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA device not available!')
    if torch.cuda.device_count() == 1:
        raise RuntimeWarning('Only 1 GPU available. Horovod is not necessary!')

    torch.backends.cudnn.benchmark = True  # Enable benchmarking for improved speed on same-sized images.

    hvd.init()  # Initialize Horovod for Pytorch
    print(f'Local Rank: {hvd.local_rank()}')

    # Set default to each GPU device. Local rank is different for each process launched by horovodrun.
    with torch.cuda.device(f'cuda:{hvd.local_rank()}'):
        main()
