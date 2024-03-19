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

def benchmark_attention(attention_module, x, num_iterations=100):
    start_time = time.time()
    for _ in range(num_iterations):
        out = attention_module(x)
        # out.backward(torch.ones_like(out))
    torch.cuda.synchronize()
    end_time = time.time()
    return (end_time - start_time) / num_iterations

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
    seq_parallel_attention = SeqParallelCrossAttention(256, seq_parallel_group=dist.group.WORLD).cuda()

    torch.manual_seed(1024)
    fast_seq_parallel_attention = FastSeqParallelCrossAttention(256, seq_parallel_group=dist.group.WORLD).cuda()
    
    torch.manual_seed(1024)
    faster_seq_parallel_attention = FasterSeqParallelCrossAttention(256, seq_parallel_group=dist.group.WORLD).cuda()
    
    torch.manual_seed(1024)
    x = torch.randn(1, 100000, 256).cuda()

    seq_x = x.clone().detach()

    x.requires_grad = True
    x.retain_grad()
    seq_x.requires_grad = True
    seq_x.retain_grad()

    
    x_list = torch.split(x, 50000, dim=1)
    sub_seq_x = x_list[rank]

    print(x.shape, sub_seq_x.shape)

    # sub_seq_x = split_forward_gather_backward(seq_x, dist.group.WORLD, dim=1, grad_scale="down")

    # run model
    out = attention(x)
    sub_seq_out = faster_seq_parallel_attention(sub_seq_x)
    sub_seq_out.mean().backward()
    out.mean().backward()
    seq_out = _gather(sub_seq_out, dim=1)

    print(out.shape, sub_seq_out.shape)

    # for i in range(5000):
    #     print(i)
    #     assert torch.allclose(seq_out[0,i,:], out[0,i,:], atol=1e-7), f"{seq_out[0,i,:]}\nvs\n{out[0,i,:]}\nvs\n{seq_out[0,i,:]-out[0,i,:]}"

    # # all reduce gradient for sp
    # for p in faster_seq_parallel_attention.parameters():
    #     if p.grad is not None:
    #         dist.all_reduce(p.grad, group=dist.group.WORLD)
    #         p.grad.div_(world_size)

    # Benchmark attention modules
    num_iterations = 10
    cross_attention_time = benchmark_attention(attention, x, num_iterations)
    seq_parallel_attention_time = benchmark_attention(seq_parallel_attention, sub_seq_x, num_iterations)
    fast_seq_parallel_attention_time = benchmark_attention(fast_seq_parallel_attention, sub_seq_x, num_iterations)
    faster_seq_parallel_attention_time = benchmark_attention(faster_seq_parallel_attention, sub_seq_x, num_iterations)

    # Print benchmark results
    print(f"CrossAttention: {cross_attention_time:.4f} seconds per iteration")
    print(f"SeqParallelCrossAttention: {seq_parallel_attention_time:.4f} seconds per iteration")
    print(f"FastSeqParallelCrossAttention: {fast_seq_parallel_attention_time:.4f} seconds per iteration")
    print(f"FasterSeqParallelCrossAttention: {faster_seq_parallel_attention_time:.4f} seconds per iteration")


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