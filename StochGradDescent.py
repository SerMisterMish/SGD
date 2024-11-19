import numpy as np
from numpy.linalg import norm
from numpy.random import choice
from numpy.typing import ArrayLike, NDArray
from typing import Callable


def StochasticGradientDescent(
    start: ArrayLike,
    X: NDArray,
    y: NDArray,
    L: Callable,
    L_grad: Callable,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    max_iter=1000,
    tol=1e-7,
    **kwargs
) -> dict:
    curr_point = start
    W_error = None

    curr_iter = 0
    while W_error is None or (curr_iter < max_iter and W_error >= tol):
        idx = choice(X.shape[0], batch_size, replace=False)
        batch_X, batch_y = X[idx, :], np.array([y[idx]]).T

        curr_value = L(curr_point, batch_X, batch_y, **kwargs)
        curr_grad = L_grad(curr_point, batch_X, batch_y, **kwargs)

        curr_point -= learning_rate * curr_grad
        W_error = norm(learning_rate * curr_grad)
        curr_iter += 1

    return {
        "point": curr_point,
        "L_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
