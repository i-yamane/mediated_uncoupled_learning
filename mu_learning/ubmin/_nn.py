from typing import Any, Union, Tuple, Callable, Optional, Sequence


from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from ..utils._adapter import SklearnAdapter
from ..utils._force2d import force2d
from ..utils._make_and_split import make_dataloader
from ..utils._make_and_split import DataTuple
from ..utils._helpers import score_UB, mse_u2y_y, mse_x2y_u2y, sce_u2y_y

import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.optim import Optimizer, Adadelta, Adam, AdamW, RMSprop, SGD  # type: ignore
from torch.optim import lr_scheduler

import mlflow

import warnings
warnings.simplefilter('default')


class NN(SklearnAdapter, BaseEstimator):  # type: ignore
    def __init__(
        self,
        model_f: 'nn.Module[torch.Tensor]',
        model_h: 'nn.Module[torch.Tensor]',
        weight_decay_f: float=1E-2,
        weight_decay_h: float=1E-2,
        n_epochs: int=500,
        w_init: float=.5,
        lr_f: float=5*1E-1,
        lr_h: float=5*1E-1,
        batch_size: int=512,
        optimizer: str='Adam',
        batch_norm: bool=False,
        grad_clip: float=1E+10,
        warm_start: bool=False,
        two_step: bool=False,
        device: Optional[torch.device]=None,
        record_loss: bool=False,
        log_metric_label: str='2StepRR'
    ) -> None:
        self.weight_decay_f = weight_decay_f
        self.weight_decay_h = weight_decay_h
        self.n_epochs = n_epochs
        self.w_init = w_init
        self.lr_f = lr_f
        self.lr_h = lr_h
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.batch_norm = batch_norm
        self.grad_clip = grad_clip
        self.record_loss = record_loss
        self.warm_start = warm_start
        self.two_step = two_step
        self.log_metric_label = log_metric_label

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.f_ = model_f
        self.f_.train()  # Enable back-propagation
        self.f_.to(self.device)

        self.h_ = model_h
        self.h_.train()  # Enable back-propagation
        self.h_.to(self.device)


    def fit_indirect(
            self,
            xu_pair: DataTuple,
            uy_pair: DataTuple
        ) -> Any:
        loader_xu = make_dataloader(xu_pair, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        loader_uy = make_dataloader(uy_pair, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.w_ = torch.tensor(self.w_init, requires_grad=False)
        self.w_.to(self.device)

        self._mse_torch = nn.MSELoss(reduction='mean')

        self._opt_f: Optimizer
        self._opt_h: Optimizer
        if self.optimizer == 'Adam':
            self._opt_f = Adam(self.f_.parameters(),
                               weight_decay=self.weight_decay_f, lr=self.lr_f)
            self._opt_h = Adam(self.h_.parameters(),
                               weight_decay=self.weight_decay_h, lr=self.lr_h)
        elif self.optimizer == 'AdamW':
            self._opt_f = AdamW(self.f_.parameters(),
                               weight_decay=self.weight_decay_f, lr=self.lr_f)
            self._opt_h = AdamW(self.h_.parameters(),
                               weight_decay=self.weight_decay_h, lr=self.lr_h)
        elif self.optimizer == 'Adadelta':
            self._opt_f = Adadelta(self.f_.parameters(),
                                   weight_decay=self.weight_decay_f,
                                   rho=0.9,
                                   lr=self.lr_f)
            self._opt_h = Adadelta(self.h_.parameters(),
                                   weight_decay=self.weight_decay_h,
                                   rho=0.9,
                                   lr=self.lr_h)
        elif self.optimizer == 'RMSprop':
            self._opt_f = RMSprop(self.f_.parameters(),
                                  weight_decay=self.weight_decay_f,
                                  lr=self.lr_f)
            self._opt_h = RMSprop(self.h_.parameters(),
                                  weight_decay=self.weight_decay_h,
                                  lr=self.lr_h)
        elif self.optimizer == 'SGD':
            self._opt_f = SGD(self.f_.parameters(),
                              weight_decay=self.weight_decay_f,
                              lr=self.lr_f,
                              momentum=0.9)
            self._opt_h = SGD(self.h_.parameters(),
                              weight_decay=self.weight_decay_h,
                              lr=self.lr_h,
                              momentum=0.9)
        else:
            raise ValueError('Specify Adam, Adadelta,'
                             'or SGD after `--optimizer`.')

        # Learning rate scheduling
        self._lr_sch_h = lr_scheduler.MultiStepLR(
            self._opt_h,
            milestones=[60, 120, 160],
            gamma=0.2
        )

        if self.warm_start or self.two_step:
            self._warm_h(loader_uy=loader_uy)
            self._warm_f(loader_xu=loader_xu)

        if not self.two_step:
            for i_epoch in range(self.n_epochs):
                for (x0, u0), (u1, y1) in zip(loader_xu, loader_uy):
                    x0, u0 = x0.to(self.device), u0.to(self.device)
                    u1, y1 = u1.to(self.device), y1.to(self.device)

                    self._update_fh(x0, u0, u1, y1, grad_clip=self.grad_clip)

                if self.record_loss:
                    lss_bat_fh = mse_x2y_u2y(
                        predict_y_from_x=self.predict_y_from_x,
                        predict_y_from_u=self.predict_y_from_u,
                        loader_xu=loader_xu,
                        device=self.device
                    )
                    lss_bat_hy = mse_u2y_y(
                        predict_y_from_u=self.predict_y_from_u,
                        loader_uy=loader_uy,
                        device=self.device
                    )
                    w = self.w_.item()
                    lss_bat_wo_BC = lss_bat_fh/w + lss_bat_hy/(-w + 1)
                    print("epch:{}|lss_bat_fh:{}".format(i_epoch, lss_bat_fh))
                    print("epch:{}|lss_bat_hy:{}".format(i_epoch, lss_bat_hy))
                    print("epch:{}|lss_bat_wo_BC:{}".format(i_epoch, lss_bat_wo_BC))
                    mlflow.log_metric(value=lss_bat_fh, key=self.log_metric_label + "_lss_bat_fh", step=i_epoch)
                    mlflow.log_metric(value=lss_bat_hy, key=self.log_metric_label + "_lss_bat_hy", step=i_epoch)
                    mlflow.log_metric(value=lss_bat_wo_BC, key=self.log_metric_label + "_lss_bat_wo_BC", step=i_epoch)

        return self

    def _warm_h(self, loader_uy):
        # type: (NN, DataLoader[Tuple[torch.Tensor, ...]]) -> None
        for i_epoch in range(self.n_epochs):
            for u1, y1 in loader_uy:
                u1, y1 = u1.to(self.device), y1.to(self.device)
                h1 = self.h_(u1)
                loss_yh_wrm = self._mse_torch(y1, h1)

                self._opt_h.zero_grad()
                loss_yh_wrm.backward()
                self._opt_h.step()

            if self.record_loss:
                lss_bat_yh_wrmh = mse_u2y_y(
                    predict_y_from_u=self.predict_y_from_u,
                    loader_uy=loader_uy,
                    device=self.device
                )
                print(i_epoch, lss_bat_yh_wrmh)
                mlflow.log_metric(
                    key=self.log_metric_label + "_lss_bat_yh_wrmh",
                    value=lss_bat_yh_wrmh,
                    step=i_epoch)

            self._lr_sch_h.step()

    def _warm_f(self, loader_xu):
        # type: (NN, DataLoader[Tuple[torch.Tensor, ...]]) -> None
        for i_epoch in range(self.n_epochs):
            for x0, u0 in loader_xu:
                x0, u0 = x0.to(self.device), u0.to(self.device)

                f0 = self.f_(x0)
                h0 = self.predict_y_from_u(u0)
                # ^^^ Calculated with eval() and no_grad.
                # ... Do not use h_().
                loss_fh = self._mse_torch(f0, h0)

                self._opt_f.zero_grad()
                loss_fh.backward()
                self._opt_f.step()

            if self.record_loss:
                lss_bat_fh_wrmf = mse_x2y_u2y(
                    predict_y_from_x=self.predict_y_from_x,
                    predict_y_from_u=self.predict_y_from_u,
                    loader_xu=loader_xu,
                    device=self.device
                )
                print(i_epoch, lss_bat_fh_wrmf)
                mlflow.log_metric(
                    key=self.log_metric_label + "_lss_bat_fh_wrmf",
                    value=lss_bat_fh_wrmf,
                    step=i_epoch)

    def _update_fh(
            self,
            x0: torch.Tensor,
            u0: torch.Tensor,
            u1: torch.Tensor,
            y1: torch.Tensor,
            grad_clip: float,
        ) -> None:
        f0 = self.f_(x0)
        h0 = self.h_(u0)
        h1 = self.h_(u1)

        loss1 = self._mse_torch(f0, h0)
        loss2 = self._mse_torch(y1, h1)

        loss = loss1/self.w_ + loss2/(-self.w_ + 1)

        self._opt_f.zero_grad()
        self._opt_h.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.f_.parameters(), grad_clip)
        torch.nn.utils.clip_grad_value_(self.h_.parameters(), grad_clip)
        self._opt_f.step()
        self._opt_h.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : array-like, shape = (n_samples, dim_x)
        Returns
        -------
        y_pred : array-like, shape = (n_samples, 1)
            Estimates of y given x. Note that it's a 2D array.
        """
        return self.predict_y_from_x(x)


    def predict_y_from_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : array-like, shape = (n_samples, dim_x)
        Returns
        -------
        y_pred : array-like, shape = (n_samples, 1)
            Estimates of y given x. Note that it's a 2D array.
        """

        check_is_fitted(self, ['f_'])

        x = force2d(x)

        self.f_.eval()
        with torch.no_grad():
            y_pred = self.f_(x)
        self.f_.train()

        return y_pred


    def predict_y_from_u(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        u : array-like, shape = (n_samples, dim_u)
        Returns
        -------
        y_pred : array-like, shape = (n_samples, 1)
            Estimates of y given u. Note that it's a 2D array.
        """
        check_is_fitted(self, ['h_'])

        u = force2d(u)

        self.h_.eval()
        with torch.no_grad():
            y_pred = self.h_(u)
        self.h_.train()

        return y_pred

