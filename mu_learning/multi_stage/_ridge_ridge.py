from typing import Any
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from .utils import Combined


def ridge_ridge_factory() -> Any:
    return Combined(
            model_x2u=Ridge(),
            model_u2y=Ridge(),
            iterative=False
        )


def kridge_kridge_factory(
            model_x2u__alpha: float=1,
            model_u2y__alpha: float=1,
            model_x2u__gamma: float=1,
            model_u2y__gamma: float=1,
            model_x2u__kernel: str='rbf',
            model_u2y__kernel: str='rbf',
        ) -> Any:
    return Combined(
            model_x2u=KernelRidge(
                    alpha=model_x2u__alpha,
                    kernel=model_x2u__kernel,
                    gamma=model_x2u__gamma,
                ),
            model_u2y=KernelRidge(
                    alpha=model_u2y__alpha,
                    kernel=model_u2y__kernel,
                    gamma=model_u2y__gamma,
                ),
            iterative=False
        )

