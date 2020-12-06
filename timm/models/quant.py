import torch

def fix_missing(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    name = 'scale'
    if name not in state_dict:
        state_dict[prefix + name] = torch.tensor(0.0, dtype=torch.float)
