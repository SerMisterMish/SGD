import numpy as np
from numpy.linalg import norm
from numpy.random import choice
from numpy.typing import NDArray
from typing import Callable


def SGD_RMSProp(
    start: NDArray,
    X: NDArray,
    y: NDArray,
    L_grad: Callable,
    batch_size: int,
    L: Callable | None = None,
    learning_rate: float = 0.01,
    decay_rate: float = 0.5,
    max_iter=1000,
    tol=1e-7,
    **kwargs
) -> dict:
    curr_point = start
    W_error = None
    run_avg = np.zeros(np.size(start))
    curr_iter = 0
    curr_value = None
    while W_error is None or (curr_iter < max_iter and W_error >= tol):
        idx = choice(X.shape[0], batch_size, replace=False)

        batch_X, batch_y = X[idx, :], np.array(y[idx]).reshape(idx.shape)

        curr_grad = L_grad(curr_point, batch_X, batch_y, **kwargs)
        run_avg = decay_rate * run_avg + (1 - decay_rate) * curr_grad**2

        curr_point -= learning_rate / np.sqrt(run_avg) * curr_grad
        W_error = norm(learning_rate * curr_grad)

        curr_iter += 1

    if L is not None:
        curr_value = L(curr_point, batch_X, batch_y, **kwargs)

    return {
        "point": curr_point,
        "L_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
