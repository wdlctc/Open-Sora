import argparse
from collections import defaultdict
from functools import reduce
import gc
import logging
import math
import operator
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
import torch.nn as nn

RPC_PORT = 29501

from open_sora.modeling.dit.attn import (
    CrossAttention,
    FastSeqParallelCrossAttention,
    FasterSeqParallelCrossAttention,
    SeqParallelCrossAttention,
)

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def _gather(input_: torch.Tensor, dim) -> torch.Tensor:

    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output

def benchmark_fsdp(rank, args, world_size):
    """Benchmark a given model using a single process and multiple devices."""
    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )
    torch.cuda.set_device(rank)
    
    torch.manual_seed(1024)
    attention = CrossAttention(256).cuda()
    
    torch.manual_seed(1024)
    seq_parallel_attention = FasterSeqParallelCrossAttention(256, seq_parallel_group=dist.group.WORLD).cuda()

    torch.manual_seed(1024)
    x = torch.randn(4, 64, 256).cuda()

    seq_x = x.clone().detach()

    x.requires_grad = True
    x.retain_grad()
    seq_x.requires_grad = True
    seq_x.retain_grad()

    
    x_list = torch.split(x, 32, dim=1)
    sub_seq_x = x_list[rank]

    print(x.shape, sub_seq_x.shape)

    # sub_seq_x = split_forward_gather_backward(seq_x, dist.group.WORLD, dim=1, grad_scale="down")

    # run model
    out = attention(x)
    sub_seq_out = seq_parallel_attention(sub_seq_x)
    seq_out = _gather(sub_seq_out, dim=1)

    print(out.shape, sub_seq_out.shape)
    
    assert torch.allclose(seq_out, out, atol=1e-7), f"{seq_out}\nvs\n{out}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local-rank', type=int, default=0)
    args, remaining = parser.parse_known_args()
    print(f"Running FSDP benchmark with args: {args}")
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(torch.cuda.device_count())
    mp.spawn(
        benchmark_fsdp,
        args=(args, num_devices),
        nprocs=num_devices,
        join=True,
    )