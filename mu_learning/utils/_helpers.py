from typing import Tuple, Any, Optional, Callable, NamedTuple
import torch
from torch._C import dtype
import torch.nn
import torch.nn.functional as F
from ..utils._adapter import UBModel
from ..utils._make_and_split import DataTuple, DataLoader
from ..utils._make_and_split import make_dataloader
from ..utils._force2d import force2d

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
        device=_device
    ) -> float:
        loader_xu = make_dataloader(data_xu, batch_size=batch_size, shuffle=True, pin_memory=True)
        loader_uy = make_dataloader(data_uy, batch_size=batch_size, shuffle=True, pin_memory=True)

        badness0: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
        n0 = len(loader_xu.dataset)
        for _, batch_xu in enumerate(loader_xu):
            x0, u0 = batch_xu

            x0 = force2d(x0)
            u0 = force2d(u0)

            y_pred_u0 = force2d(model.predict_y_from_u(u0))
            y_pred_x0 = force2d(model.predict_y_from_x(x0))

            with torch.no_grad():
                badness0 += (y_pred_u0 - y_pred_x0).pow(2).sum() / (-w + 1) / n0

        badness1: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
        n1 = len(loader_uy.dataset)
        for _, batch_uy in enumerate(loader_uy):
            u1, y1 = batch_uy

            u1 = force2d(u1)
            y1 = force2d(y1)

            y_pred_u1 = force2d(model.predict_y_from_u(u1))

            with torch.no_grad():
                badness1 += (y1 - y_pred_u1).pow(2).sum() / w / n1

        return -(badness0 + badness1).item()


def mse_x2u_u(predict_u_from_x, loader_xu, device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_u_from_x: Function to predict u from x.
    loader_xu: Data loader of data tuples of x and u.
    device: Backend device for torch.
    """
    loss_sum: float = 0
    n = 0
    for x, u in loader_xu:
        x, u = x.to(device), u.to(device).float()
        u_pred = predict_u_from_x(x).to(device)
        loss_sum += F.mse_loss(u_pred, u, reduction='sum').item()
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
    for u, y in loader_uy:
        u, y = u.to(device), y.to(device).float()
        output = predict_y_from_u(u).to(device)
        # F.cross_entropy applies log-softmax to the first argument.
        loss_sum += F.cross_entropy(
            output,
            y.argmax(dim=1),
            reduction='sum').item()
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
    for u, y in loader_uy:
        u, y = u.to(device), y.to(device).float()
        output = predict_y_from_u(u).to(device)
        loss_sum += F.mse_loss(output, y, reduction='sum').item()
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
    for x, u in loader_xu:
        x, u = x.to(device), u.to(device).float()
        f_pred = predict_y_from_x(x).to(device)
        h_pred = predict_y_from_u(u).to(device)
        loss_sum += F.mse_loss(f_pred, h_pred, reduction='sum').item()
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
    for x, y in loader_xy:
        x, y = x.to(device), y.to(device).float()
        output = predict_y_from_x(x).to(device)
        loss_sum += F.mse_loss(output, y, reduction='sum').item()
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


class Losses_UB(NamedTuple):
    upper_loss_total: float
    lower_loss_total: float
    loss1: float
    loss2: float
    loss3_factor1: float
    loss3_factor2: float


def kl_CSUB_fullbatch(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        loader_uy,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> Losses_UB
    """
    predict_y_from_x: Function to predict y from x.
    predict_y_from_u: Function to predict y from u.
    loader_xy: Data loader of data tuples of x and y.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.

    """
    return kl_UB_fullbatch_helper(
        predict_y_from_x=predict_y_from_x,
        predict_y_from_u=predict_y_from_u,
        loader_xu=loader_xu,
        loader_uy=loader_uy,
        UB_type='CSUB',
        device=device)


def kl_SumUB_fullbatch(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        loader_uy,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> Losses_UB
    """
    predict_y_from_x: Function to predict y from x.
    predict_y_from_u: Function to predict y from u.
    loader_xy: Data loader of data tuples of x and y.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.
    """
    return kl_UB_fullbatch_helper(
        predict_y_from_x=predict_y_from_x,
        predict_y_from_u=predict_y_from_u,
        loader_xu=loader_xu,
        loader_uy=loader_uy,
        UB_type='SumUB',
        device=device)


def kl_UB_fullbatch_helper(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        loader_uy,
        UB_type,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], DataLoader[Tuple[torch.Tensor, ...]], str, Optional[Any]) -> Losses_UB
    """
    predict_y_from_x: Function to predict y from x.
    predict_y_from_u: Function to predict y from u.
    loader_xy: Data loader of data tuples of x and y.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.
    """
    loss1_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    loss2_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    loss3_factor1_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    loss3_factor2_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    n0 = 0

    for x0, u0 in loader_xu:
        x0, u0 = x0.to(device), u0.to(device)
        f0 = predict_y_from_x(x0).to(device)
        h0 = predict_y_from_u(u0).to(device)
        # KL-divergence with soft-max. F.kl_div(..., log_targets=True)inputs log_probabilities:
        loss1_sum += F.kl_div(
            h0.log_softmax(dim=1),
            f0.log_softmax(dim=1),
            log_target=True,
            reduction='sum'
        )

        loss3_factor2_sum += F.mse_loss(h0.log_softmax(dim=1), f0.log_softmax(dim=1), reduction='sum')

        n0 += x0.size(0)
    loss1 = loss1_sum / n0
    loss3_factor2 = loss3_factor2_sum / n0

    n1 = 0
    for u1, y1 in loader_uy:
        u1, y1 = u1.to(device), y1.to(device).float()
        h1 = predict_y_from_u(u1).to(device)

        loss2_sum += F.cross_entropy(h1, y1.argmax(dim=1), reduction='sum')
        loss3_factor1_sum += F.mse_loss(y1, h1.log_softmax(dim=1), reduction='sum')

        n1 += u1.size(0)
    loss2 = loss2_sum / n1
    loss3_factor1 = loss3_factor1_sum / n1

    upper_loss_total: Optional[torch.Tensor]
    lower_loss_total: Optional[torch.Tensor]
    if UB_type == 'CSUB':
        upper_loss_total = loss1 + loss2 + loss3_factor1.sqrt() * loss3_factor2.sqrt()
        lower_loss_total = loss1 + loss2 - loss3_factor1.sqrt() * loss3_factor2.sqrt()
    elif UB_type == 'SumUB':
        upper_loss_total = loss1 + loss2 + (loss3_factor1 + loss3_factor2) / 2
        lower_loss_total = loss1 + loss2 - (loss3_factor1 + loss3_factor2) / 2
    else:
        raise ValueError('Invalid value for UB_type: {}. Pass `CSUB` or `SumUB`.'.format(UB_type))

    assert upper_loss_total != None
    assert lower_loss_total != None

    return Losses_UB(
        upper_loss_total=upper_loss_total.item(),
        lower_loss_total=lower_loss_total.item(),
        loss1=loss1.item(),
        loss2=loss2.item(),
        loss3_factor1=loss3_factor1.item(),
        loss3_factor2=loss3_factor2.item()
    )


def kldiv_x2y_u2y(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> float
    """
    predict_y_from_x: Function to predict y from x before applying softmax.
    predict_y_from_u: Function to predict y from u before applying softmax.
    loader_xy: Data loader of data tuples of x and y.
    device: Backend device for torch.
    """
    loss_fh_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    n = 0

    for x0, u0 in loader_xu:
        x0, u0 = x0.to(device), u0.to(device)
        f0 = predict_y_from_x(x0).to(device)
        h0 = predict_y_from_u(u0).to(device)
        # KL-divergence with soft-max. F.kl_div(..., log_target=True) inputs
        # log-probabilities and compares them:
        loss_fh_sum += F.kl_div(
            h0.log_softmax(dim=1),
            f0.log_softmax(dim=1),
            log_target=True,
            reduction='sum'
        )
        n += x0.size(0)

    return loss_fh_sum.item() / n


def l2_CSUB_fullbatch(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        loader_uy,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> Losses_UB
    return l2_UB_fullbatch_helper(
        predict_y_from_x=predict_y_from_x,
        predict_y_from_u=predict_y_from_u,
        loader_xu=loader_xu,
        loader_uy=loader_uy,
        UB_type='CSUB',
        device=device)


def l2_SumUB_fullbatch(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        loader_uy,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], DataLoader[Tuple[torch.Tensor, ...]], Optional[Any]) -> Losses_UB
    return l2_UB_fullbatch_helper(
        predict_y_from_x=predict_y_from_x,
        predict_y_from_u=predict_y_from_u,
        loader_xu=loader_xu,
        loader_uy=loader_uy,
        UB_type='SumUB',
        device=device)


def l2_UB_fullbatch_helper(
        predict_y_from_x,
        predict_y_from_u,
        loader_xu,
        loader_uy,
        UB_type,
        device=_device):
    # type: (Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], DataLoader[Tuple[torch.Tensor, ...]], DataLoader[Tuple[torch.Tensor, ...]], str, Optional[Any]) -> Losses_UB
    """
    predict_y_from_x: Function to predict y from x.
    predict_y_from_u: Function to predict y from u.
    loader_xy: Data loader of data tuples of x and y.
    loader_uy: Data loader of data tuples of u and y.
    device: Backend device for torch.
    """
    loss1_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    loss2_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    loss3_factor1_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)
    loss3_factor2_sum: torch.Tensor = torch.tensor(0.0, dtype=torch.float64, requires_grad=False).to(device)

    n0 = 0
    for x0, u0 in loader_xu:
        x0, u0 = x0.to(device), u0.to(device)
        f0 = predict_y_from_x(x0).to(device)
        h0 = predict_y_from_u(u0).to(device)

        loss1_sum += F.mse_loss(h0, f0, reduction='sum')
        loss3_factor2_sum += F.mse_loss(h0, f0, reduction='sum')

        n0 += x0.size(0)
    loss1: torch.Tensor = loss1_sum / n0
    loss3_factor2: torch.Tensor = loss3_factor2_sum / n0

    n1 = 0
    for u1, y1 in loader_uy:
        u1, y1 = u1.to(device), y1.to(device).float()
        h1 = predict_y_from_u(u1).to(device)

        loss2_sum += F.mse_loss(h1, y1, reduction='sum')
        loss3_factor1_sum += F.mse_loss(y1, h1, reduction='sum')

        n1 += u1.size(0)
    loss2: torch.Tensor = loss2_sum / n1
    loss3_factor1: torch.Tensor = loss3_factor1_sum / n1

    upper_loss_total: Optional[torch.Tensor]
    lower_loss_total: Optional[torch.Tensor]
    if UB_type == 'CSUB':
        upper_loss_total = loss1 + loss2 + loss3_factor1.sqrt() * loss3_factor2.sqrt()
        lower_loss_total = loss1 + loss2 - loss3_factor1.sqrt() * loss3_factor2.sqrt()
    elif UB_type == 'SumUB':
        upper_loss_total = loss1 + loss2 + (loss3_factor1 + loss3_factor2) / 2
        lower_loss_total = loss1 + loss2 - (loss3_factor1 + loss3_factor2) / 2
    else:
        raise ValueError('Invalid value for UB_type: {}. Pass `CSUB` or `SumUB`.'.format(UB_type))

    assert upper_loss_total != None
    assert lower_loss_total != None

    return Losses_UB(
        upper_loss_total=upper_loss_total.item(),
        lower_loss_total=lower_loss_total.item(),
        loss1=loss1.item(),
        loss2=loss2.item(),
        loss3_factor1=loss3_factor1.item(),
        loss3_factor2=loss3_factor2.item()
    )
