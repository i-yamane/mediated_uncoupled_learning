from typing import Dict, Tuple, Sequence, Any, Union, Optional
from typing_extensions import Protocol

from abc import abstractmethod
from abc import ABCMeta
import torch
from ._make_and_split import split_x_or_u
from ._make_and_split import split_u_or_y
from ._make_and_split import DataTuple


class SklearnAdapter(metaclass=ABCMeta):
    dim_x: int
    dim_u: int

    @abstractmethod
    def fit_indirect(
            self,
            data_xu: DataTuple,
            data_uy: DataTuple
        ) -> Any:
        pass

    @abstractmethod
    def score_indirect(
            self,
            data_xu: DataTuple,
            data_uy: DataTuple,
            sample_weight: Optional[Sequence[float]]=None
        ) -> float:
        pass

    def score(
            self,
            x_or_u: torch.Tensor,
            u_or_y: torch.Tensor,
            sample_weight: Optional[Sequence[float]]=None
        ) -> float:
        self.check_is_initialized(self, ['dim_x', 'dim_u'])
        x0, u1 = split_x_or_u(x_or_u, dim_x=self.dim_x)
        u0, y1 = split_u_or_y(u_or_y, dim_u=self.dim_u)
        return self.score_indirect((x0, u0), (u1, y1))

    def fit(
            self,
            x_or_u: torch.Tensor,
            u_or_y: torch.Tensor
        ) -> Any:
        self.check_is_initialized(self, ['dim_x', 'dim_u'])
        x0, u1 = split_x_or_u(x_or_u, dim_x=self.dim_x)
        u0, y1 = split_u_or_y(u_or_y, dim_u=self.dim_u)
        return self.fit_indirect((x0, u0), (u1, y1))

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def check_is_initialized(instance: Any, attrnames: Sequence[str]) -> None:
        for name in attrnames:
            if not hasattr(instance, name):
                raise AttributeError('Set {} before calling fit()'.format(name))


class SklearnAdapterMat(metaclass=ABCMeta):
    dim_x: int
    dim_u: int

    @abstractmethod
    def fit_indirect(
            self,
            xu_pair: Tuple[torch.Tensor, torch.Tensor],
            uy_pair: Tuple[torch.Tensor, torch.Tensor]
        ) -> Any:
        pass

    @abstractmethod
    def score_indirect(
            self,
            xu_pair: Tuple[torch.Tensor, torch.Tensor],
            uy_pair: Tuple[torch.Tensor, torch.Tensor],
            sample_weight: Optional[Sequence[float]]=None
        ) -> float:
        pass

    def score(
            self,
            x_or_u: torch.Tensor,
            u_or_y: torch.Tensor,
            sample_weight: Optional[Sequence[float]]=None
        ) -> float:
        self.check_is_initialized(self, ['dim_x', 'dim_u'])
        x0, u1 = split_x_or_u(x_or_u, dim_x=self.dim_x)
        u0, y1 = split_u_or_y(u_or_y, dim_u=self.dim_u)
        return self.score_indirect((x0, u0), (u1, y1))

    def fit(
            self,
            x_or_u: torch.Tensor,
            u_or_y: torch.Tensor
        ) -> Any:
        self.check_is_initialized(self, ['dim_x', 'dim_u'])
        x0, u1 = split_x_or_u(x_or_u, dim_x=self.dim_x)
        u0, y1 = split_u_or_y(u_or_y, dim_u=self.dim_u)
        return self.fit_indirect((x0, u0), (u1, y1))

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def check_is_initialized(instance: Any, attrnames: Sequence[str]) -> None:
        for name in attrnames:
            if not hasattr(instance, name):
                raise AttributeError('Set {} before calling fit()'.format(name))


class Predict(Protocol):
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        ...


class SklearnModel(Predict, Protocol):
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> Any:
        ...

    def partial_fit(self, x: torch.Tensor, y: torch.Tensor) -> Any:
        ...


class PredictUFromX(Protocol):
    def predict_u_from_x(self, x: torch.Tensor) -> torch.Tensor:
        ...


class PredictYFromU(Protocol):
    def predict_y_from_u(self, u: torch.Tensor) -> torch.Tensor:
        ...


class PredictYFromX(Protocol):
    def predict_y_from_x(self, x: torch.Tensor) -> torch.Tensor:
        ...


class UBModel(PredictYFromU, PredictYFromX, Protocol):
    pass

