import numpy as np
from numpy.linalg import norm
from numpy.random import choice
from numpy.typing import NDArray
from typing import Callable


def SGD_RMSProp(
    start: NDArray,
    X: NDArray,
    y: NDArray,
    L: Callable,
    L_grad: Callable,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    decay_rate: float = 0.5,
    max_iter=1000,
    tol=1e-7,
    **kwargs
) -> dict:
    curr_point = start
    W_error = None
    run_avg = np.zeros(np.size(start))
    curr_iter = 0
    while W_error is None or (curr_iter < max_iter and W_error >= tol):
        idx = choice(X.shape[0], batch_size, replace=False)
        batch_X, batch_y = X[idx, :], np.array([y[idx]]).reshape(idx.shape)

        curr_value = L(curr_point, batch_X, batch_y, **kwargs)
        curr_grad = L_grad(curr_point, batch_X, batch_y, **kwargs)
        run_avg = decay_rate * run_avg + (1 - decay_rate) * curr_grad**2

        curr_point -= learning_rate / np.sqrt(run_avg) * curr_grad
        W_error = norm(learning_rate * curr_grad)
        curr_iter += 1

    return {
        "point": curr_point,
        "L_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
