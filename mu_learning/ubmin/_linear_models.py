from typing import Any, Optional, Sequence, Tuple
import torch
from torch.nn.functional import mse_loss
from sklearn.base import BaseEstimator
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_is_fitted
from ..utils._adapter import SklearnAdapterMat
from ..utils._force2d import force2d
from ..utils._make_and_split import DataTuple

import warnings
warnings.simplefilter('default')


class LinearModel(SklearnAdapterMat, BaseEstimator):  # type: ignore
    def __init__(
            self,
            fit_method: str,
            dim_x: int,
            dim_u: int,
            reg_level_f: float=1E-2,
            reg_level_h: float=1E-2,
            band_width_f: float=1,
            band_width_h: float=1,
            b_f: int=1000,
            b_h: int=1000,
            base_fn: str='rbf',
            w: float=.1,
            device: Optional[torch.device]=None
        ) -> None:
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.reg_level_f = reg_level_f
        self.reg_level_h = reg_level_h
        self.band_width_f = band_width_f
        self.band_width_h = band_width_h
        self.b_f = b_f
        self.b_h = b_h
        self.base_fn = base_fn
        self.w = w
        self.fit_method = fit_method

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def fit_indirect(
            self,
            xu_pair: DataTuple,
            uy_pair: DataTuple
        ) -> Any:
        if self.fit_method == 'analytic_primal':
            return self.fit_indirect_primal(xu_pair, uy_pair)
        elif self.fit_method == 'analytic_dual':
            return self.fit_indirect_dual(xu_pair, uy_pair)

    def fit_indirect_primal(
            self,
            xu_pair: DataTuple,
            uy_pair: DataTuple
        ) -> Any:
        # TODO: test
        # TODO: Use all the training data. Currently, it only uses one mini-batch.
        x0, u0 = list(xu_pair)[0]
        u1, y1 = list(uy_pair)[0]
        x0, u0 = x0.to(self.device), u0.to(self.device)
        u1, y1 = u1.to(self.device), y1.to(self.device)

        self.n0_ = u0.shape[0]
        self.n1_ = u1.shape[0]

        if self.base_fn == 'linear':
            phi0 = x0
            psi0 = u0
            psi1 = u1
        elif self.base_fn == 'rbf':
            # Kernel centers:
            self.vx_ = x0
            self.vu_ = torch.cat([u0, u1], dim=0)

            phi0 = gaussian_kernel(x0, self.vx_, band_width=self.band_width_f)
            psi0 = gaussian_kernel(u0, self.vu_, band_width=self.band_width_h)
            psi1 = gaussian_kernel(u1, self.vu_, band_width=self.band_width_h)
        else:
            raise ValueError('base_fn must be either \'linear or rbf\'')

        phipsi0 = torch.cat((phi0, -psi0), axis=1)
        zeropsi1 = torch.cat((torch.zeros(psi1.shape[0], phi0.shape[1]).to(self.device), psi1), axis=1)
        A = phipsi0.T @ phipsi0 / (self.n0_ * self.w) \
            + zeropsi1.T @ zeropsi1 / (self.n1_ * (1 - self.w))
        b = torch.cat((torch.zeros((1, phi0.shape[1])).to(self.device), y1.T @ psi1), axis=1) \
            / (self.n1_ * (1 - self.w))
        th, _ = torch.solve(b.T, A)
        self.alpha_ = th[:phi0.shape[1], :]
        self.beta_ = th[phi0.shape[1]:(phi0.shape[1] + psi1.shape[1]), :]

        return self

    def fit_indirect_dual(
            self,
            xu_pair: DataTuple,
            uy_pair: DataTuple
        ) -> Any:
        x0, u0 = list(xu_pair)[0]
        u1, y1 = list(uy_pair)[0]
        x0, u0 = x0.to(self.device), u0.to(self.device)
        u1, y1 = u1.to(self.device), y1.to(self.device)
        #x0 = force2d(x0)
        #u0 = force2d(u0)
        #u1 = force2d(u1)
        #y1 = force2d(y1)

        self.n0_ = u0.shape[0]
        self.n1_ = u1.shape[0]

        if self.base_fn == 'linear':
            phi0 = x0
            psi0 = u0
            psi1 = u1
        elif self.base_fn == 'rbf':
            # Kernel centers:
            self.vx_ = x0
            self.vu_ = torch.cat([u0, u1], dim=0)

            phi0 = gaussian_kernel(x0, self.vx_, band_width=self.band_width_f)
            psi0 = gaussian_kernel(u0, self.vu_, band_width=self.band_width_h)
            psi1 = gaussian_kernel(u1, self.vu_, band_width=self.band_width_h)
        else:
            raise ValueError('base_fn must be either \'linear or rbf\'')

        M1 = (phi0.T @ phi0) / (self.n0_ * self.w) \
            + self.reg_level_f * torch.eye(phi0.shape[1]).to(self.device) / self.n0_
        M2 = (phi0.T @ psi0) / (self.n0_ * self.w)
        M3 = (psi0.T @ psi0) / (self.n0_ * self.w) \
             + (psi1.T @ psi1) / (self.n1_ * (1 - self.w)) \
             + self.reg_level_h * torch.eye(psi0.shape[1]).to(self.device) / (self.n0_ + self.n1_)
        invM1 = torch.inverse(M1)
        invM =  torch.inverse(M3 - M2.T @ invM1 @ M2)
        b1 = (psi1.T @ y1) / (self.n1_ * (1 - self.w))

        self.beta_ = force2d(invM @ b1)
        self.alpha_ = - force2d(invM1 @ M2 @ invM @ b1)

        # For DEBUGGING:
        # score = self.score_indirect(xu_pair, uy_pair)
        y_pred_x0 = force2d(self.predict_y_from_x(x0))
        y_pred_u0 = force2d(self.predict_y_from_u(u0))
        y_pred_u1 = force2d(self.predict_y_from_u(u1))

        with torch.no_grad():
            badness = mse_loss(y_pred_x0, y_pred_u0) / self.w \
                      + mse_loss(y_pred_u1, y1) / (1 - self.w)

        print(badness)

        return self

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
        check_is_fitted(self, ['alpha_'])
        x = force2d(x)
        if self.base_fn == 'linear':
            phi = x
        elif self.base_fn == 'rbf':
            phi = gaussian_kernel(x, self.vx_, band_width=self.band_width_f)
        else:
            raise ValueError('base_fn must be either \'linear or rbf\'')

        return force2d(phi @ self.alpha_)

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
        check_is_fitted(self, ['beta_'])
        u = force2d(u)
        if self.base_fn == 'linear':
            psi = u
        elif self.base_fn == 'rbf':
            psi = gaussian_kernel(u, self.vu_, band_width=self.band_width_h)
        else:
            raise ValueError('base_fn must be either \'linear or rbf\'')

        # return force2d(psi.dot(self.beta_))
        return force2d(psi @ self.beta_)

    def score_indirect(
            self,
            xu_pair: Tuple[torch.Tensor, torch.Tensor],
            uy_pair: Tuple[torch.Tensor, torch.Tensor],
            sample_weight: Optional[Sequence[float]]=None
        ) -> float:
        """Returns the NEGATIVE objective values calculated with indirect data:
            - 2 * ((y_pred_x - y_pred_u) ** 2).mean()
            - 2 * (y_pred_u - y_true) ** 2).mean()
        The negative of this is an estimated upper bound of the mean square error (MSE).
        It can be smaller than the MSE due to its estimation error,
        but the estimator is unbiased.
        The smaller it is, the better it is.
        It is intended to be used for validation without ordinary data
        of (x, y) pairs.
        It is recommended to use sklearn.metrics.mean_squared_error()
        when such ordinary data are available.
        Parameters
        ----------
        x_or_u : array-like, shape = (n_samples, dim_x + dim_u + 1)
            Validation instances of either x or u.
            x_or_u[i, :dim_x] represents an instance of x if x_or_u[i, -1] == 0
            x_or_u[i, dim_x:-1] represents an instance of u if x_or_u[i, -1] == 1
        u_or_y : array-like, shape = (n_samples, dim_u + 1 + 1)
            Labels of either u or y.
            u_or_y[i, :dim_u] represents the label of u for the instance x_or_u[i, :dim_x] if u_or_y[i, -1] == 0.
            u_or_y[i, dim_u] represents the label of y for the instance x_or_u[i, dim_x:-1] if u_or_y[i, -1] == 1.
            In fact, u_or_y[i, -1] == x_or_u[i, -1].
        sample_weight : array-like, shape = [n_samples], optional
            Sample weights.
        Returns
        -------
        score : float
            An estimated upper bound of the mean square error of self.predict(x).
        """

        x0, u0 = list(xu_pair)[0]
        u1, y1 = list(uy_pair)[0]
        x0, u0 = x0.to(self.device), u0.to(self.device)
        u1, y1 = u1.to(self.device), y1.to(self.device)

        y_pred_x0 = force2d(self.predict_y_from_x(x0))
        y_pred_u0 = force2d(self.predict_y_from_u(u0))
        y_pred_u1 = force2d(self.predict_y_from_u(u1))

        with torch.no_grad():
            badness = mse_loss(y_pred_x0, y_pred_u0) / self.w \
                      + mse_loss(y_pred_u1, y1) / (1 - self.w)
        return -badness.item()


def gaussian_kernel(x: torch.Tensor, v: torch.Tensor, band_width: float) -> torch.Tensor:
    vx = x @ v.T
    xx = (x**2).sum(dim=1)
    vv = (v**2).sum(dim=1)
    distmat = xx[:, None] - 2 * vx + vv[None, :]
    phi = torch.exp(- distmat / band_width)
    return phi
