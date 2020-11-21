import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Demo")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    print(args)

    gpu_per_node = torch.cuda.device_count()

    # This the global rank: 0, 1, 2, ..., 15
    global_rank = int(os.environ['RANK'])
    print("int(os.environ['RANK']) = %d" % global_rank)

    # This the globak world_size
    world_size = int(os.environ['WORLD_SIZE'])
    print("world_size = %d" % world_size)

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    local_rank = args.local_rank
    print(f"Running basic DDP example on local rank {local_rank}.")

    # create model and move it to GPU with id rank
    device = torch.device("cuda:" + str(local_rank))

    model = ToyModel().to(device)
    if global_rank == 0:
        print(model)

    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(device)
    loss = loss_fn(outputs, labels)
    print("rank=%d, loss=%f" % (local_rank, loss))
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()

