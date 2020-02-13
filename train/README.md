# Training with multiple processes.

---
Concepts used in Horovod and MPI are outlined [here](https://horovod.readthedocs.io/en/latest/concepts_include.html).

---

When `horovodrun` is used, multiple Python processes are launched simultaneously.

These processes run almost independently of one another, though they do communicate to pass data.

Each process is given an identifier, the *"local rank"*.

Each process can access its identifier via the hvd.local_rank() function.

This is similar to how CUDA threads operate.

Imagine launching the `python ...` command simultaneously, in parallel, 
but each process knowing its identifying number.

Note that the DataLoaders launch new workers in each process.

This means that the number of pre-processing processes 
is multiplied by the number of Horovod processes.

This may cause memory issues or performance drops.

The mini-batch size is not affected by the number of Horovod processes because of the DistributedSampler.

The Horovod documentation recommends that only model checkpoints and logs from 
"hvd.local_rank() == 0" should be saved.

The Ring-AllReduce will ensure that the values will converge eventually.

Within each `horovodrun` process, the device assigned to that process is set as the default device.

Alternatively, one could simple send all tensors in that process to `torch.device(f'cuda:{hvd.local_rank()}')`.

---
That's it! Have fun coding your Pytorch projects with Horovod.

You can simply copy the docker environment from here and change the training code.

You will also probably have to change some run settings.

__*Please star/fork my repository if you find my explanations helpful!*__ 