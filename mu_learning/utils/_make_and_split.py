from typing import Dict, Tuple, Sequence, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from ._force2d import force2d


DataTuple = None


def split_x_or_u(
        x_or_u: torch.Tensor,
        dim_x: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    select = {
            'x': x_or_u[:, -1] == 0,
            'u': x_or_u[:, -1] == 1,
        }

    x = x_or_u[select['x'], :dim_x]
    u = x_or_u[select['u'], dim_x:-1]

    return x, u


def split_u_or_y(
        u_or_y: torch.Tensor,
        dim_u: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    select = {
            'u': u_or_y[:, -1] == 0,
            'y': u_or_y[:, -1] == 1
        }

    u = u_or_y[select['u'], :dim_u]
    y = u_or_y[select['y'], dim_u]

    return u, y


def make_x_or_u(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return torch.cat([
            torch.cat([x, torch.zeros([x.shape[0], u.shape[1]]), torch.zeros([x.shape[0], 1])], dim=1),
            torch.cat([torch.zeros([u.shape[0], x.shape[1]]), u, torch.ones([u.shape[0], 1])], dim=1),
        ], dim=0)


def make_u_or_y(u: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if len(y.shape) == 1:
        y = y[:, None]
    return torch.cat([
            torch.cat([u, torch.zeros([u.shape[0], y.shape[1]]), torch.zeros([u.shape[0], 1])], dim=1),
            torch.cat([torch.zeros([y.shape[0], u.shape[1]]), y, torch.ones([y.shape[0], 1])], dim=1),
        ], dim=0)


def make_dataloader(data, batch_size, shuffle, pin_memory):
    # type: (DataTuple, int, bool, bool) -> DataLoader[Tuple[torch.Tensor, ...]]
    if isinstance(data, DataLoader):
        return data
    elif isinstance(data, Dataset):
        dataset = data
    elif isinstance(data, tuple) and len(data) == 2 \
            and isinstance(data[0], torch.Tensor) \
            and isinstance(data[1], torch.Tensor):
        dataset = TensorDataset(data[0], data[1])
    # elif isinstance(data, tuple) and len(tuple) == 2 \
    #         and isinstance(data[0], np.ndarray) \
    #         and isinstance(data[1], np.ndarray):
    #     tensor1 = torch.from_numpy(force2d(data[0])).float()
    #     tensor2 = torch.from_numpy(force2d(data[1])).float()
    #     dataset = TensorDataset(tensor1, tensor2)
    else:
        raise ValueError("make_dataloader only accepts torch.utils.data.DataLoader, torch.utils.data.Dataset, or a pair of torch.Tensor.") 

    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      pin_memory=pin_memory)

