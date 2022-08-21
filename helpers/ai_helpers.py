import torch


def count_model_params(model, verbose = True):     

  num_params = (sum(p.numel() for p in model.parameters())) / 1e6
  if verbose:
    print(f"model has {num_params:.4f} Million params")
  return num_params
