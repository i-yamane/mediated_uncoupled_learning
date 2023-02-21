import abc
from typing import Dict, Tuple, Sequence, Any, Union, Optional
import torch
from torch import nn
from torch.optim import Optimizer, Adadelta, Adam, AdamW, RMSprop, SGD  # type: ignore
from sklearn.base import BaseEstimator
# from sklearn.utils.validation import check_is_fitted
# from sklearn.metrics import mean_squared_error as mse_sk
import mlflow
from ...utils._adapter import SklearnAdapter, SklearnModel
from ...utils._force2d import force2d
from ...utils._make_and_split import make_dataloader
from ...utils._make_and_split import DataTuple
from ...utils._helpers import mse_u2y_y, mse_x2u_u, sce_u2y_y


class Combined(SklearnAdapter, BaseEstimator):  # type: ignore
    _mse = nn.MSELoss(reduction='mean')
    def __init__(
        self,
        model_x2u: 'nn.Module[torch.Tensor]',
        model_u2y: 'nn.Module[torch.Tensor]',
        n_epochs: int=100,
        batch_size: int=512,
        lr_x2u: float=1E-1,
        lr_u2y: float=1E-1,
        grad_clip: float=1E+10,
        weight_decay: float=0,
        optimizer: str='Adadelta',
        lossfn_u2y: str = 'squared',
        device: Optional[torch.device]=None,
        record_loss: bool=False,
        log_metric_label: str='CMB'
    ) -> None:
        self.model_x2u = model_x2u
        self.model_u2y = model_u2y
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_x2u = lr_x2u
        self.lr_u2y = lr_u2y
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.lossfn_u2y = lossfn_u2y
        self.device = device
        self.record_loss = record_loss
        self.log_metric_label = log_metric_label

        self.model_x2u.to(device)
        self.model_u2y.to(device)

    def fit_indirect(self, data_xu: DataTuple, data_uy: DataTuple) -> Any:
        if not isinstance(self.model_x2u, nn.Module) \
                and not isinstance(self.model_u2y, nn.Module):
            raise ValueError('Expected type torch.nn.Module'
                             'for model_x2u and model.u2y.')

        self._mse_torch = nn.MSELoss(reduction='mean')
        self._sce_torch = nn.CrossEntropyLoss(reduction='mean')

        self._opt_x2u: Optimizer
        self._opt_u2y: Optimizer
        if self.optimizer == 'Adam':
            self._opt_x2u = Adam(self.model_x2u.parameters(),
                                 weight_decay=self.weight_decay, lr=self.lr_x2u)
            self._opt_u2y = Adam(self.model_u2y.parameters(),
                                 weight_decay=self.weight_decay, lr=self.lr_u2y)
        elif self.optimizer == 'AdamW':
            self._opt_x2u = AdamW(self.model_x2u.parameters(),
                                 weight_decay=self.weight_decay, lr=self.lr_x2u)
            self._opt_u2y = AdamW(self.model_u2y.parameters(),
                                 weight_decay=self.weight_decay, lr=self.lr_u2y)
        elif self.optimizer == 'Adadelta':
            self._opt_x2u = Adadelta(self.model_x2u.parameters(), rho=0.9,
                                     weight_decay=self.weight_decay, lr=self.lr_x2u)
            self._opt_u2y = Adadelta(self.model_u2y.parameters(), rho=0.9,
                                     weight_decay=self.weight_decay, lr=self.lr_u2y)
        elif self.optimizer == 'RMSprop':
            self._opt_x2u = RMSprop(self.model_x2u.parameters(),
                                    weight_decay=self.weight_decay, lr=self.lr_x2u)
            self._opt_u2y = RMSprop(self.model_u2y.parameters(),
                                    weight_decay=self.weight_decay, lr=self.lr_u2y)
        elif self.optimizer == 'SGD':
            self._opt_x2u = SGD(self.model_x2u.parameters(), lr=self.lr_x2u,
                                weight_decay=self.weight_decay)
            self._opt_u2y = SGD(self.model_u2y.parameters(), lr=self.lr_u2y,
                                weight_decay=self.weight_decay)
        else:
            raise ValueError('Specify Adam, Adadelta, or SGD'
                             'for optimizer option.')

        loader_xu = make_dataloader(data_xu, batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True)
        loader_uy = make_dataloader(data_uy, batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True)

        for i_epoch in range(self.n_epochs):
            for i_batch, (xu_batch, uy_batch) in enumerate(zip(loader_xu, loader_uy)):
                step = i_epoch * self.batch_size + i_batch

                x0, u0 = xu_batch
                x0, u0 = x0.to(self.device), u0.to(self.device)
                u0_pred = self.model_x2u(x0)
                loss_x2u = self._mse_torch(u0, u0_pred)

                u1, y1 = uy_batch
                u1, y1 = u1.to(self.device), y1.to(self.device)
                y1_pred = self.model_u2y(u1)
                if self.lossfn_u2y == 'squared':
                    loss_u2y = self._mse_torch(y1, y1_pred)
                elif self.lossfn_u2y == 'softmax_cross_entropy':
                    loss_u2y = self._sce_torch(y1_pred, y1.argmax(dim=1))
                else:
                    raise ValueError('Pass \'squared\' or \'softmax_cross_entropy\'')

                self._opt_x2u.zero_grad()
                loss_x2u.backward()
                torch.nn.utils.clip_grad_value_(self.model_x2u.parameters(), self.grad_clip)
                self._opt_x2u.step()

                self._opt_u2y.zero_grad()
                loss_u2y.backward()
                torch.nn.utils.clip_grad_value_(self.model_u2y.parameters(), self.grad_clip)
                self._opt_u2y.step()

            if self.record_loss:
                if self.lossfn_u2y == 'squared':
                    lss_bat_u2y = mse_u2y_y(
                        predict_y_from_u=self.model_u2y,
                        loader_uy=loader_uy,
                        device=self.device
                    )
                elif self.lossfn_u2y == 'softmax_cross_entropy':
                    lss_bat_u2y = sce_u2y_y(
                        predict_y_from_u=self.model_u2y,
                        loader_uy=loader_uy,
                        device=self.device
                    )
                else:
                    raise ValueError('Pass \'squared\' or \'softmax_cross_entropy\'')
                print(i_epoch, lss_bat_u2y)
                mlflow.log_metric(
                    key=self.log_metric_label + "_lss_bat_u2y",
                    value=lss_bat_u2y,
                    step=i_epoch)
                lss_bat_x2u = mse_x2u_u(
                    predict_u_from_x=self.model_x2u,
                    loader_xu=loader_xu,
                    device=self.device
                )
                print(i_epoch, lss_bat_x2u)
                mlflow.log_metric(
                    key=self.log_metric_label + "_lss_bat_x2u",
                    value=lss_bat_x2u,
                    step=i_epoch)


    def predict_y_from_x(self, x: torch.Tensor) -> torch.Tensor:
        x = force2d(x)
        u_pred = self.predict_u_from_x(x)
        return force2d(self.predict_y_from_u(u_pred)).float()

    def predict_u_from_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : array-like, shape = (n_samples, dim_x)
        Returns
        -------
        u_pred : array-like, shape = (n_samples, 1)
            Estimates of u given x. Note that it's a 2D array.
        """

        x = force2d(x)

        self.model_x2u.eval()
        with torch.no_grad():
            u_pred = self.model_x2u(x)
        self.model_x2u.train()

        return u_pred

    def predict_y_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u : array-like, shape = (n_samples, dim_x)
        Returns
        -------
        y_pred : array-like, shape = (n_samples, 1)
            Estimates of y given u. Note that it's a 2D array.
        """

        u = force2d(u)

        self.model_u2y.eval()
        with torch.no_grad():
            y_pred = self.model_u2y(u)
        self.model_u2y.train()

        return y_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_y_from_x(x)

    def score_indirect(
        self,
        data_xu: DataTuple,
        data_uy: DataTuple,
        # sample_weight: Optional[Sequence[float]]=None
    ) -> float:
        loader_xu = make_dataloader(data_xu, batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True)
        loader_uy = make_dataloader(data_uy, batch_size=self.batch_size,
                                    shuffle=True, pin_memory=True)

        badness0 = torch.tensor(0)
        n0 = len(loader_xu.dataset)
        for i_batch, batch_xu in enumerate(loader_xu):
            x0, u0 = batch_xu
            x0 = force2d(x0)
            u0 = force2d(u0)
            u0_pred = force2d(self.predict_u_from_x(x0))
            with torch.no_grad():
                badness0 += (u0 - u0_pred).square().sum() / n0

        badness1 = torch.tensor(0)
        n1 = len(loader_uy.dataset)
        for i_batch, batch_uy in enumerate(loader_uy):
            u1, y1 = batch_uy
            u1 = force2d(u1)
            y1 = force2d(y1)
            y1_pred = force2d(self.predict_y_from_u(u1))
            with torch.no_grad():
                badness1 += (y1 - y1_pred).square().sum() / n1

        return -(badness0 + badness1).item()

