from typing import Any, Tuple, Optional, Sequence

from sklearn.base import BaseEstimator
from ..utils._adapter import SklearnAdapter
from ..utils._force2d import force2d
from ..utils._make_and_split import make_dataloader
from ..utils._make_and_split import DataTuple
from ..utils._helpers import mse_u2y_y
from ..utils._helpers import mse_x2y_u2y
from ..utils._helpers import sce_u2y_y
from ..utils._helpers import kl_CSUB_fullbatch
from ..utils._helpers import kl_SumUB_fullbatch
from ..utils._helpers import kldiv_x2y_u2y
from ..utils._helpers import l2_CSUB_fullbatch
from ..utils._helpers import l2_SumUB_fullbatch
from ..utils._helpers import Losses_UB

import torch
import torch.nn
import torch.nn.functional as F
import torch.cuda
from torch.utils.data import DataLoader
from torch.optim import Optimizer, Adadelta, Adam, AdamW, RMSprop, SGD  # type: ignore
from torch.optim import lr_scheduler

import mlflow

import warnings
warnings.simplefilter('default')


class NN(SklearnAdapter, BaseEstimator):  # type: ignore
    def __init__(
        self,
        model_f: 'torch.nn.Module[torch.Tensor]',
        model_h: 'torch.nn.Module[torch.Tensor]',
        weight_decay_f: float = 0,
        weight_decay_h: float = 0,
        n_epochs: int = 200,
        w_init: float = .5,
        lr_f: float = 1E-2,
        lr_h: float = 1E-2,
        batch_size: int = 512,
        optimizer: str = 'Adam',
        batch_norm: bool = False,
        grad_clip: float = 1E+10,
        warm_start: bool = True,
        two_step: bool = False,
        loss_type: str = 'cross_entropy_CSUB',
        device: Optional[torch.device] = None,
        record_loss: bool = False,
        log_metric_label: str = 'JointBregMU_CE'
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
        self.loss_type = loss_type
        self.log_metric_label = log_metric_label

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.f_ = model_f
        self.h_ = model_h
    def fit_indirect(
            self,
            xu_pair: DataTuple,
            uy_pair: DataTuple
        ) -> Any:
        """ Fit the model to (x, u)-pairs and (u, y)-pairs in an MU-learning setup.
        """

        self.f_.train()
        self.h_.train()
        self.f_.to(self.device)
        self.h_.to(self.device)

        loader_xu = make_dataloader(xu_pair, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        loader_uy = make_dataloader(uy_pair, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self.w_ = torch.tensor(self.w_init, requires_grad=False)
        self.w_.to(self.device)

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
            raise ValueError('Specify'
                             ' Adam'
                             ', Adadelta'
                             ', SGD'
                             ', or RMSprop'
                             'after `--optimizer` option.')

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
            # # Reset the learning rates
            # for param_group in self._opt_f.param_groups:
            #     param_group['lr'] = self.lr_f
            # for param_group in self._opt_h.param_groups:
            #     param_group['lr'] = self.lr_h

            for i_epoch in range(self.n_epochs):
                for (x0, u0), (u1, y1) in zip(loader_xu, loader_uy):
                    x0, u0 = x0.to(self.device), u0.to(self.device)
                    u1, y1 = u1.to(self.device), y1.to(self.device)

                    self._update_fh(x0, u0, u1, y1, grad_clip=self.grad_clip)

                if self.record_loss:
                    if self.loss_type == 'cross_entropy_heuristic':
                        loss_fh_batch = mse_x2y_u2y(
                            predict_y_from_x=self.predict_y_from_x,
                            predict_y_from_u=self.predict_y_from_u,
                            loader_xu=loader_xu,
                            device=self.device
                        )
                        print("epch:{}|loss_fh_batch:{}".format(i_epoch, loss_fh_batch))
                        mlflow.log_metric(
                            value=loss_fh_batch,
                            key=self.log_metric_label + "_loss_fh_batch",
                            step=i_epoch
                        )

                        loss_yh_batch = sce_u2y_y(
                            predict_y_from_u=self.predict_y_from_u,
                            loader_uy=loader_uy,
                            device=self.device
                        )
                        print("epch:{}|loss_yh_batch:{}".format(i_epoch, loss_yh_batch))
                        mlflow.log_metric(
                            value=loss_yh_batch,
                            key=self.log_metric_label + "_loss_yh_batch",
                            step=i_epoch
                        )
                    elif self.loss_type in ['cross_entropy_CSUB', 'cross_entropy_SumUB']:
                        loss_fh_batch = kldiv_x2y_u2y(
                            predict_y_from_x=self.predict_y_from_x,
                            predict_y_from_u=self.predict_y_from_u,
                            loader_xu=loader_xu,
                            device=self.device
                        )
                        print("epch:{}|loss_fh_batch:{}".format(i_epoch, loss_fh_batch))
                        mlflow.log_metric(
                            value=loss_fh_batch,
                            key=self.log_metric_label + "_loss_fh_batch",
                            step=i_epoch
                        )

                        loss_yh_batch = sce_u2y_y(
                            predict_y_from_u=self.predict_y_from_u,
                            loader_uy=loader_uy,
                            device=self.device
                        )
                        print("epch:{}|loss_yh_batch:{}".format(i_epoch, loss_yh_batch))
                        mlflow.log_metric(
                            value=loss_yh_batch,
                            key=self.log_metric_label + "_loss_yh_batch",
                            step=i_epoch
                        )

                        if self.loss_type == 'cross_entropy_CSUB':
                            losses_UB_batch = kl_CSUB_fullbatch(
                                predict_y_from_x=self.predict_y_from_x,
                                predict_y_from_u=self.predict_y_from_u,
                                loader_xu=loader_xu,
                                loader_uy=loader_uy,
                                device=self.device
                            )
                        elif self.loss_type == 'cross_entropy_SumUB':
                            losses_UB_batch = kl_SumUB_fullbatch(
                                predict_y_from_x=self.predict_y_from_x,
                                predict_y_from_u=self.predict_y_from_u,
                                loader_xu=loader_xu,
                                loader_uy=loader_uy,
                                device=self.device
                            )

                        print("epch:{}|loss_total_UB_batch:{}".format(i_epoch, losses_UB_batch.upper_loss_total))
                        mlflow.log_metric(
                            value=losses_UB_batch.upper_loss_total,
                            key=self.log_metric_label + "_loss_total_UB_batch",
                            step=i_epoch
                        )
                        print("epch:{}|loss1_UB_batch:{}".format(i_epoch, losses_UB_batch.loss1))
                        mlflow.log_metric(
                            value=losses_UB_batch.loss1,
                            key=self.log_metric_label + "_loss1_UB_batch",
                            step=i_epoch
                        )
                        print("epch:{}|loss2_UB_batch:{}".format(i_epoch, losses_UB_batch.loss2))
                        mlflow.log_metric(
                            value=losses_UB_batch.loss2,
                            key=self.log_metric_label + "_loss2_UB_batch",
                            step=i_epoch
                        )
                        print("epch:{}|loss3_factor1_UB_batch:{}".format(i_epoch, losses_UB_batch.loss3_factor1))
                        mlflow.log_metric(
                            value=losses_UB_batch.loss3_factor1,
                            key=self.log_metric_label + "_loss3_factor1_UB_batch",
                            step=i_epoch
                        )
                        print("epch:{}|loss3_factor2_UB_batch:{}".format(i_epoch, losses_UB_batch.loss3_factor2))
                        mlflow.log_metric(
                            value=losses_UB_batch.loss3_factor2,
                            key=self.log_metric_label + "_loss3_factor2_UB_batch",
                            step=i_epoch
                        )
                    elif self.loss_type in ['l2_CSUB', 'l2_SumUB']:
                        loss_fh_batch = mse_x2y_u2y(
                            predict_y_from_x=self.predict_y_from_x,
                            predict_y_from_u=self.predict_y_from_u,
                            loader_xu=loader_xu,
                            device=self.device
                        )
                        print("epch:{}|loss_fh_batch:{}".format(i_epoch, loss_fh_batch))
                        mlflow.log_metric(
                            value=loss_fh_batch,
                            key=self.log_metric_label + "_loss_fh_batch",
                            step=i_epoch
                        )

                        loss_yh_batch = mse_u2y_y(
                            predict_y_from_u=self.predict_y_from_u,
                            loader_uy=loader_uy,
                            device=self.device
                        )
                        w = self.w_.item()
                        print("epch:{}|loss_yh_batch:{}".format(i_epoch, loss_yh_batch))
                        mlflow.log_metric(value=loss_yh_batch, key=self.log_metric_label + "_loss_yh_batch", step=i_epoch)

                        losses_UB_batch: Optional[Losses_UB]
                        if self.loss_type == 'l2_CSUB':
                            losses_UB_batch = l2_CSUB_fullbatch(
                                predict_y_from_x=self.predict_y_from_x,
                                predict_y_from_u=self.predict_y_from_u,
                                loader_xu=loader_xu,
                                loader_uy=loader_uy,
                                device=self.device
                            )
                        elif self.loss_type == 'l2_SumUB':
                            losses_UB_batch = l2_SumUB_fullbatch(
                                predict_y_from_x=self.predict_y_from_x,
                                predict_y_from_u=self.predict_y_from_u,
                                loader_xu=loader_xu,
                                loader_uy=loader_uy,
                                device=self.device
                            )

                        assert(losses_UB_batch != None)

                        print("epch:{}|{}_loss_UB_total_batch:{}".format(
                            i_epoch,
                            self.log_metric_label,
                            losses_UB_batch.upper_loss_total))
                        mlflow.log_metric(value=losses_UB_batch.upper_loss_total,
                                          key=self.log_metric_label + "_loss_UB_total_batch",
                                          step=i_epoch)
                        print("epch:{}|{}_loss1_UB_batch:{}".format(
                            i_epoch,
                            self.log_metric_label,
                            losses_UB_batch.loss1
                        ))
                        mlflow.log_metric(value=losses_UB_batch.loss1,
                                          key=self.log_metric_label + "_loss1_UB_batch",
                                          step=i_epoch)
                        print("epch:{}|{}_loss2_UB_batch:{}".format(
                            i_epoch,
                            self.log_metric_label,
                            losses_UB_batch.loss2))
                        mlflow.log_metric(value=losses_UB_batch.loss2,
                                          key=self.log_metric_label + "_loss2_UB_batch",
                                          step=i_epoch)
                        print("epch:{}|{}_loss3_factor1_UB_batch:{}".format(
                            i_epoch,
                            self.log_metric_label,
                            losses_UB_batch.loss3_factor1))
                        mlflow.log_metric(value=losses_UB_batch.loss3_factor1,
                                          key=self.log_metric_label + "_loss3_factor1_UB_batch",
                                          step=i_epoch)
                        print("epch:{}|{}_loss3_factor2_UB_batch:{}".format(
                            i_epoch,
                            self.log_metric_label,
                            losses_UB_batch.loss3_factor2))
                        mlflow.log_metric(value=losses_UB_batch.loss3_factor2,
                                          key=self.log_metric_label + "_loss3_factor2_UB_batch",
                                          step=i_epoch)
                    else:
                        loss_fh_batch = mse_x2y_u2y(
                            predict_y_from_x=self.predict_y_from_x,
                            predict_y_from_u=self.predict_y_from_u,
                            loader_xu=loader_xu,
                            device=self.device
                        )
                        print("epch:{}|loss_fh_batch:{}".format(i_epoch, loss_fh_batch))
                        mlflow.log_metric(
                            value=loss_fh_batch,
                            key=self.log_metric_label + "_loss_fh_batch",
                            step=i_epoch
                        )

                        loss_yh_batch = mse_u2y_y(
                            predict_y_from_u=self.predict_y_from_u,
                            loader_uy=loader_uy,
                            device=self.device
                        )
                        w = self.w_.item()
                        print("epch:{}|loss_yh_batch:{}".format(i_epoch, loss_yh_batch))
                        mlflow.log_metric(value=loss_yh_batch, key=self.log_metric_label + "_loss_yh_batch", step=i_epoch)

                        loss_UB_batch = loss_fh_batch/w + loss_yh_batch/(-w + 1)
                        print("epch:{}|loss_UB_batch:{}".format(i_epoch, loss_UB_batch))
                        mlflow.log_metric(value=loss_UB_batch, key=self.log_metric_label + "_loss_UB_batch", step=i_epoch)

        return self

    def fit_direct(
            self,
            xy_pair: DataTuple
        ) -> Any:
        """ Fit the model to (x, y)-pairs in a usual supervised learning setup.
        """

        self.f_.train()
        self.f_.to(self.device)

        loader_xy = make_dataloader(xy_pair, batch_size=self.batch_size, shuffle=True, pin_memory=True)

        self._opt_f: Optimizer
        if self.optimizer == 'Adam':
            self._opt_f = Adam(self.f_.parameters(),
                               weight_decay=self.weight_decay_f, lr=self.lr_f)
        elif self.optimizer == 'AdamW':
            self._opt_f = AdamW(self.f_.parameters(),
                               weight_decay=self.weight_decay_f, lr=self.lr_f)
        elif self.optimizer == 'Adadelta':
            self._opt_f = Adadelta(self.f_.parameters(),
                                   weight_decay=self.weight_decay_f,
                                   rho=0.9,
                                   lr=self.lr_f)
        elif self.optimizer == 'RMSprop':
            self._opt_f = RMSprop(self.f_.parameters(),
                                  weight_decay=self.weight_decay_f,
                                  lr=self.lr_f)
        elif self.optimizer == 'SGD':
            self._opt_f = SGD(self.f_.parameters(),
                              weight_decay=self.weight_decay_f,
                              lr=self.lr_f,
                              momentum=0.9)
        else:
            raise ValueError('Specify'
                             ' Adam'
                             ', Adadelta'
                             ', SGD'
                             ', or RMSprop'
                             'after `--optimizer` option.')

        # Learning rate scheduling
        self._lr_sch_f = lr_scheduler.MultiStepLR(
            self._opt_f,
            milestones=[60, 120, 160],
            gamma=0.2
        )

        for i_epoch in range(self.n_epochs):
            for x1, y1 in loader_xy:
                x1, y1 = x1.to(self.device), y1.to(self.device)
                f1 = self.f_(x1)
                if self.loss_type in ['cross_entropy', 'cross_entropy_CSUB', 'cross_entropy_SumUB', 'cross_entropy_heuristic']:
                    loss_yf = F.cross_entropy(f1, y1.argmax(dim=1), reduction='mean')
                elif self.loss_type in ['l2', 'l2_CSUB', 'l2_SumUB', 'JointRR']:
                    loss_yf = F.mse_loss(y1, f1, reduction='mean')
                else:
                    raise ValueError('Invalid loss_type: {}. Specify cross_entropy_CSUB,'
                                    'cross_entropy_heuristic, or l2_CSUB'.format(self.loss_type))

                self._opt_f.zero_grad()
                loss_yf.backward()
                self._opt_f.step()

            if self.record_loss:
                if self.loss_type in [
                        'cross_entropy',
                        'cross_entropy_CSUB',
                        'cross_entropy_SumUB',
                        'cross_entropy_heuristic']:
                    loss_yf_batch = sce_u2y_y(
                        predict_y_from_u=self.predict_y_from_x,
                        loader_uy=loader_xy,
                        device=self.device
                    )
                elif self.loss_type in ['l2', 'l2_CSUB', 'l2_SumUB', 'JointRR']:
                    loss_yf_batch = mse_u2y_y(
                        predict_y_from_u=self.predict_y_from_u,
                        loader_uy=loader_xy,
                        device=self.device
                    )
                else:
                    raise ValueError('Invalid loss_type: {}. Specify cross_entropy_CSUB,'
                                    'cross_entropy_heuristic, or l2_CSUB'.format(self.loss_type))

                print("epch:{}|loss_yf_batch:{}".format(
                    i_epoch, loss_yf_batch))
                mlflow.log_metric(
                    key=self.log_metric_label + "_loss_yf_batch",
                    value=loss_yf_batch,
                    step=i_epoch)

            self._lr_sch_f.step()

    def _warm_h(self, loader_uy):
        # type: (NN, DataLoader[Tuple[torch.Tensor, ...]]) -> None
        for i_epoch in range(self.n_epochs):
            for u1, y1 in loader_uy:
                u1, y1 = u1.to(self.device), y1.to(self.device)
                h1 = self.h_(u1)
                if self.loss_type in ['cross_entropy', 'cross_entropy_CSUB', 'cross_entropy_SumUB', 'cross_entropy_heuristic']:
                    loss_yh_wrm = F.cross_entropy(h1, y1.argmax(dim=1), reduction='mean')
                elif self.loss_type in ['l2', 'l2_CSUB', 'l2_SumUB', 'JointRR']:
                    loss_yh_wrm = F.mse_loss(y1, h1, reduction='mean')
                else:
                    raise ValueError('Invalid loss_type: {}. Specify cross_entropy_CSUB,'
                                     'cross_entropy_heuristic, or l2_CSUB'.format(self.loss_type))

                self._opt_h.zero_grad()
                loss_yh_wrm.backward()
                self._opt_h.step()

            if self.record_loss:
                if self.loss_type in ['cross_entropy', 'cross_entropy_CSUB', 'cross_entropy_SumUB', 'cross_entropy_heuristic']:
                    loss_yh_batch_warm_h = sce_u2y_y(
                        predict_y_from_u=self.predict_y_from_u,
                        loader_uy=loader_uy,
                        device=self.device
                    )
                elif self.loss_type in ['l2', 'l2_CSUB', 'l2_SumUB', 'JointRR']:
                    loss_yh_batch_warm_h = mse_u2y_y(
                        predict_y_from_u=self.predict_y_from_u,
                        loader_uy=loader_uy,
                        device=self.device
                    )
                else:
                    raise ValueError('Invalid loss_type: {}. Specify cross_entropy_CSUB,'
                                     'cross_entropy_heuristic, or l2_CSUB'.format(self.loss_type))

                print("epch:{}|loss_yh_batch_warm_h:{}".format(
                    i_epoch, loss_yh_batch_warm_h))
                mlflow.log_metric(
                    key=self.log_metric_label + "_loss_yh_batch_warm_h",
                    value=loss_yh_batch_warm_h,
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

                if self.loss_type == 'cross_entropy_heuristic':
                    loss_fh = F.cross_entropy(f0, h0.argmax(dim=1), reduction='mean')
                elif self.loss_type in ['cross_entropy', 'cross_entropy_CSUB', 'cross_entropy_SumUB']:
                    # KL-divergence with soft-max:
                    loss_fh = F.kl_div(
                        h0.log_softmax(dim=1),
                        f0.log_softmax(dim=1),
                        log_target=True,
                        reduction='batchmean'
                    )
                elif self.loss_type in ['l2', 'l2_CSUB', 'l2_SumUB', 'JointRR']:
                    loss_fh = F.mse_loss(f0, h0, reduction='mean')
                else:
                    raise ValueError('Invalid loss_type: {}.'
                                     ' Specify cross_entropy_CSUB,'
                                     ' cross_entropy_SumUB,'
                                     ' cross_entropy_heuristic,'
                                     ' or l2_CSUB'.format(self.loss_type))

                self._opt_f.zero_grad()
                loss_fh.backward()
                self._opt_f.step()

            if self.record_loss:
                if self.loss_type == 'cross_entropy_heuristic':
                    loss_fh_batch_warm_f = mse_x2y_u2y(
                        predict_y_from_x=self.predict_y_from_x,
                        predict_y_from_u=self.predict_y_from_u,
                        loader_xu=loader_xu,
                        device=self.device
                    )
                elif self.loss_type in ['cross_entropy', 'cross_entropy_CSUB', 'cross_entropy_SumUB']:
                    loss_fh_batch_warm_f = kldiv_x2y_u2y(
                        predict_y_from_x=self.predict_y_from_x,
                        predict_y_from_u=self.predict_y_from_u,
                        loader_xu=loader_xu,
                        device=self.device
                    )
                elif self.loss_type in ['l2', 'l2_CSUB', 'l2_SumUB', 'JointRR']:
                    loss_fh_batch_warm_f = mse_x2y_u2y(
                        predict_y_from_x=self.predict_y_from_x,
                        predict_y_from_u=self.predict_y_from_u,
                        loader_xu=loader_xu,
                        device=self.device
                    )
                else:
                    raise ValueError('Invalid loss_type: {}.'
                                     ' Specify cross_entropy_CSUB,'
                                     ' cross_entropy_SumUB,'
                                     ' cross_entropy_heuristic,'
                                     ' or l2_CSUB'.format(self.loss_type))

                print("epch:{}|loss_fh_batch_warm_f:{}".format(
                    i_epoch, loss_fh_batch_warm_f))
                mlflow.log_metric(
                    value=loss_fh_batch_warm_f,
                    key=self.log_metric_label + "_loss_fh_batch_warm_f",
                    step=i_epoch
                )

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

        loss :torch.Tensor
        if self.loss_type == 'cross_entropy_heuristic':
            loss1 = F.cross_entropy(f0, h0.argmax(dim=1), reduction='mean')
            loss2 = F.cross_entropy(h1, y1.argmax(dim=1), reduction='mean')
            loss = loss1/self.w_ + loss2/(-self.w_ + 1)
        elif self.loss_type in ['cross_entropy_CSUB', 'cross_entropy_SumUB']:
            # KL-divergence with soft-max:
            loss1 = F.kl_div(
                h0.log_softmax(dim=1),
                f0.log_softmax(dim=1),
                log_target=True,
                reduction='batchmean'  # Use this for the standard KL.
            )
            # ...This expects log-probabilities for input.
            # ...It also expects log-probabilities for target with log_target=True.

            loss2 = F.cross_entropy(h1, y1.argmax(dim=1), reduction='mean')

            loss3_factor1 = F.mse_loss(y1, h1.softmax(dim=1), reduction='mean')
            loss3_factor2 = F.mse_loss(h0.log_softmax(dim=1), f0.log_softmax(dim=1), reduction='mean')

            loss3: Optional[torch.Tensor] = None
            if self.loss_type == 'cross_entropy_CSUB':
                loss3 = loss3_factor1.sqrt() * loss3_factor2.sqrt()
            elif self.loss_type == 'cross_entropy_SumUB':
                loss3 = (loss3_factor1 + loss3_factor2) / 2
            if loss3 == None:
                raise RuntimeError('loss3 got None, but we have no clue on the cause...')

            loss = loss1 + loss2 + loss3
        elif self.loss_type in ['l2_CSUB', 'l2_SumUB']:
            loss1 = F.mse_loss(h0, f0, reduction='mean')
            loss2 = F.mse_loss(h1, y1, reduction='mean')

            loss3_factor1 = F.mse_loss(y1, h1, reduction='mean')
            loss3_factor2 = F.mse_loss(h0, f0, reduction='mean')

            loss3: Optional[torch.Tensor] = None
            if self.loss_type == 'l2_CSUB':
                loss3 = loss3_factor1.sqrt() * loss3_factor2.sqrt()
            elif self.loss_type == 'l2_SumUB':
                loss3 = (loss3_factor1 + loss3_factor2) / 2

            if loss3 == None:
                raise RuntimeError('loss3 got None, but we have no clue on the cause...')

            loss = loss1 + loss2 + loss3
        elif self.loss_type == 'JointRR':
            loss1 = F.mse_loss(f0, h0, reduction='mean')
            loss2 = F.mse_loss(y1, h1, reduction='mean')
            loss = loss1/self.w_ + loss2/(-self.w_ + 1)
        else:
            raise ValueError('Invalid loss_type: {}.'
                                ' Specify cross_entropy_CSUB,'
                                ' cross_entropy_SumUB,'
                                ' cross_entropy_heuristic,'
                                ' JointRR,'
                                ' or l2_CSUB'.format(self.loss_type))

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
        u = force2d(u)

        self.h_.eval()
        with torch.no_grad():
            y_pred = self.h_(u)
        self.h_.train()

        return y_pred

    def score_indirect(
            self,
            data_xu: DataTuple,
            data_uy: DataTuple,
            sample_weight: Optional[Sequence[float]]=None
        ) -> float:
        raise NotImplementedError
