from typing import Dict, Tuple, Sequence, Any, Union, Optional, Callable
import torch
from ..utils._adapter import UBModel
from ..utils._make_and_split import DataTuple, DataLoader
from ..utils._make_and_split import make_dataloader
from ..utils._force2d import force2d
from ..utils._adapter import Predict, PredictYFromU

if torch.cuda.is_available():  # type:ignore
    _device = torch.device('cuda')
else:
    _device = torch.device('cpu')


def score_UB(
        model: UBModel,
        data_xu: DataTuple,
        data_uy: DataTuple,
        batch_size: int,
        w: float,
        sample_weight: Optional[Sequence[float]]=None
    ) -> float:
        loader_xu = make_dataloader(data_xu, batch_size=batch_size, shuffle=True, pin_memory=True)
        loader_uy = make_dataloader(data_uy, batch_size=batch_size, shuffle=True, pin_memory=True)

        badness0 = torch.tensor(0)
        n0 = len(loader_xu.dataset)
        for i_batch, batch_xu in enumerate(loader_xu):
            x0, u0 = batch_xu

            x0 = force2d(x0)
            u0 = force2d(u0)

            y_pred_u0 = force2d(model.predict_y_from_u(u0))
            y_pred_x0 = force2d(model.predict_y_from_x(x0))

            with torch.no_grad():
                badness0 += (y_pred_u0 - y_pred_x0).pow(2).sum() / (-w + 1) / n0

        badness1 = torch.tensor(0)
        n1 = len(loader_uy.dataset)
        for i_batch, batch_uy in enumerate(loader_uy):
            u1, y1 = batch_uy

            u1 = force2d(u1)
            y1 = force2d(y1)

            y_pred_u1 = force2d(model.predict_y_from_u(u1))

            with torch.no_grad():
                badness1 += (y1 - y_pred_u1).pow(2).sum() / w / n1

        return -(badness0 + badness1).item()


def mse_x2u_u(predict_u_from_x, loader_xu, device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> torch.Tensor
    """
    predict_u_from_x: Function to predict u from x.
    loader_xu: Data loader of data tuples of x and u.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    calc_mse = torch.nn.MSELoss(reduction='sum')
    for x, u in loader_xu:
        x, u = x.to(device), u.to(device).float()
        u_pred = predict_u_from_x(x).to(device)
        loss_sum += calc_mse(u_pred, u).item()
        n += u.size(0)

    return loss_sum / n


def sce_u2y_y(predict_y_from_u, loader_uy, device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_u: Function to predict y from u.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    calc_ce = torch.nn.CrossEntropyLoss(reduction='sum')
    for u, y in loader_uy:
        u, y = u.to(device), y.to(device).float()
        output = predict_y_from_u(u).to(device)
        loss_sum += calc_ce(output, y.argmax(dim=1)).item()
        n += y.size(0)

    return loss_sum / n


def mse_u2y_y(predict_y_from_u, loader_uy, device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_u: Function to predict y from u.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    calc_mse = torch.nn.MSELoss(reduction='sum')
    for u, y in loader_uy:
        u, y = u.to(device), y.to(device).float()
        output = predict_y_from_u(u).to(device)
        loss_sum += calc_mse(output, y).item()
        n += y.size(0)

    return loss_sum / n


def mse_x2y_u2y(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_u: Function to predict y from u.
    predict_y_from_x: Function to predict y from x.
    loader_xu: Data loader of data tuples of x and u.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    calc_mse = torch.nn.MSELoss(reduction='sum')
    for x, u in loader_xu:
        x, u = x.to(device), u.to(device).float()
        f_pred = predict_y_from_x(x).to(device)
        h_pred = predict_y_from_u(u).to(device)
        loss_sum += calc_mse(f_pred, h_pred).item()
        n += u.size(0)

    return loss_sum / n


def mse_x2y_y(
        predict_y_from_x,
        loader_xy,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_u: Function to predict y from u.
    predict_y_from_x: Function to predict y from x.
    loader_xu: Data loader of data tuples of x and u.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    calc_mse = torch.nn.MSELoss(reduction='sum')
    for x, y in loader_xy:
        x, y = x.to(device), y.to(device).float()
        output = predict_y_from_x(x).to(device)
        loss_sum += calc_mse(output, y).item()
        n += y.size(0)

    return loss_sum / n


def acc_u2y_y(predict_y_from_u, loader_uy, device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_u: Function to predict y from u.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    for u, y in loader_uy:
        u, y = u.to(device), y.to(device).float()
        outputs = predict_y_from_u(u).to(device)
        _, y_pred = torch.max(outputs, dim=1)
        _, y_num = torch.max(y, dim=1)
        loss_sum += y_pred.eq(y_num).sum().item()
        n += y.size(0)

    return loss_sum / n


def acc_x2y_u2y(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_u: Function to predict y from u.
    predict_y_from_x: Function to predict y from x.
    loader_xu: Data loader of data tuples of x and u.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    for x, u in loader_xu:
        x, u = x.to(device), u.to(device).float()
        outputs_f = predict_y_from_x(x).to(device)
        outputs_h = predict_y_from_u(u).to(device)
        _, y_pred_f = torch.max(outputs_f, dim=1)
        _, y_pred_h = torch.max(outputs_h, dim=1)
        loss_sum += y_pred_f.eq(y_pred_h).sum().item()
        n += u.size(0)

    return loss_sum / n


def acc_x2y_y(
        predict_y_from_x,
        loader_xy,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_x: Function to predict y from x.
    loader_xu: Data loader of data tuples of x and u.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    for x, y in loader_xy:
        x, y = x.to(device), y.to(device).float()
        outputs = predict_y_from_x(x).to(device)
        _, y_pred = torch.max(outputs, dim=1)
        _, y_num = torch.max(y, dim=1)
        loss_sum += y_pred.eq(y_num).sum().item()
        n += y.size(0)

    return loss_sum / n
