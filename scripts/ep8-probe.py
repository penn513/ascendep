import os
from datetime import timedelta
import torch
import torch_npu
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
assert torch.npu.device_count() == 8
torch.npu.set_device(rank)
dist.init_process_group("hccl", timeout=timedelta(seconds=90))
x = torch.arange(8, device="npu", dtype=torch.bfloat16) + rank * 8
y = torch.empty_like(x)
dist.all_to_all_single(y, x)
assert torch.equal(y.cpu(), torch.arange(8) * 8 + rank)
a = torch.ones(8, 8, device="npu", dtype=torch.bfloat16, requires_grad=True)
(a @ a).sum().backward()
assert (a.grad.cpu() == 16).all()
print(f"rank={rank} BF16 backward and HCCL AllToAll passed", flush=True)
dist.destroy_process_group()
