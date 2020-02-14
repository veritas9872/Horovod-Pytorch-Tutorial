# Training with multiple processes.

Concepts used in Horovod and MPI are outlined [here](https://horovod.readthedocs.io/en/latest/concepts_include.html).

### How does it work?

When `horovodrun` is used, multiple Python processes are launched simultaneously.

These processes run almost independently of one another, though they do communicate to pass data.

Each process is given an identifier, the *"local rank"*.

Each process can access its identifier via the hvd.local_rank() function.

This is similar to how CUDA threads operate within a CUDA kernel.

Imagine launching the `python ...` command simultaneously, in parallel, 
but each process knowing its identifying number.

Each process does its own thing but synchronizes with the others via the Ring-AllReduce.

### How many processes? 

As mentioned previously, note that the DataLoaders launch new workers in each process.

This means that the number of pre-processing processes 
is multiplied by the number of Horovod processes.

This may cause memory issues or performance drops.

However, **mini-batch size is not affected by the number of Horovod processes.**
 
DistributedSampler handles this very well.

### How to write logs and save checkpoints.

The Horovod documentation recommends that only model checkpoints and logs from 
`"hvd.local_rank() == 0"` should be saved.

The Ring-AllReduce ensures that the different versions will not diverge very much.

### How to manage devices

Within each `horovodrun` process, the device assigned to that process is set as the default device.

Alternatively, one could simple send all tensors in that process to `torch.device(f'cuda:{hvd.local_rank()}')`.

---
That's it! Have fun coding your Pytorch projects with Horovod.

You can simply copy the docker environment from here and change the training code.

You will also probably have to change some run settings.

__*Please star/fork my repository if you find my explanations helpful!*__ 