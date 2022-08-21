import torch
from pkg_resources import packaging
import torch.cuda.nccl as nccl
import torch.distributed as dist

def check_bfloat_support(check_network_support=True, verbose=True):
  """ verify hardware support for bfloat16 - gpu and network """
  gpu_support = torch.version.cuda and torch.cuda.is_bf16_supported()
  network_support = packaging.version.parse(torch.version.cuda).release >= (11, 0) and dist.is_nccl_available() and nccl.version() >= (2, 10)

  if verbose:
    print(f"gpu bfloat support = {gpu_support}")
    if check_network_support:
      print(f"network bfloat support = {network_support}")
    
  return gpu_support if not check_network_support else (gpu_support and network_support)

def count_model_params(model, verbose = True): 
  """ count total model parameters """

  num_params = (sum(p.numel() for p in model.parameters())) / 1e6
  if verbose:
    print(f"model has {num_params:.4f} Million params")
  return num_params
