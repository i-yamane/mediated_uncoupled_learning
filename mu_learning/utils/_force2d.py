import torch
import numpy as np  # type: ignore

def force2d(ary: torch.Tensor) -> torch.Tensor:
    # Note that numpy does not have type info
    if isinstance(ary, np.ndarray):
        ary = torch.from_numpy(ary)
    elif isinstance(ary, torch.Tensor):
        pass
    else:
        raise ValueError('Expected type numpy.ndarray or torch.Tensor for ary')

    if len(ary.shape) == 1:
        return ary[:, None]
    return ary

