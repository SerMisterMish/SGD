import numpy as np
from numpy.linalg import norm
from numpy.random import choice
from numpy.typing import NDArray
from typing import Callable


def StochasticGradientDescent(
    start: NDArray,
    X: NDArray,
    y: NDArray,
    L_grad: Callable,
    batch_size: int,
    L: Callable | None = None,
    learning_rate: float = 0.01,
    max_iter=1000,
    tol=1e-4,
    **kwargs
) -> dict:
    curr_point = start
    W_error = None
    curr_value = None
    curr_iter = 0
    while W_error is None or (curr_iter < max_iter and W_error >= tol):
        idx = choice(X.shape[0], batch_size, replace=False)
        batch_X, batch_y = X[idx, :], np.array(y[idx]).reshape(idx.shape)

        curr_grad = L_grad(curr_point, batch_X, batch_y, **kwargs)

        curr_point -= learning_rate * curr_grad
        W_error = norm(learning_rate * curr_grad)
        curr_iter += 1

    if L is not None:
        curr_value = L(curr_point, X, y, **kwargs)
    return {
        "point": curr_point,
        "L_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
