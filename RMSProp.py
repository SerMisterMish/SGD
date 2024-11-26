import numpy as np
from numpy.linalg import norm
from numpy.random import choice, permutation
from numpy.typing import NDArray
from typing import Callable


def SGD_RMSProp(
    start: NDArray,
    X: NDArray,
    y: NDArray,
    L_grad: Callable,
    batch_size: int | float,
    L: Callable | None = None,
    learning_rate: float = 0.01,
    decay_rate: float = 0.5,
    use_epoch: bool = False,
    max_iter=1000,
    tol=1e-7,
    n_iter_no_change: int = 5,
    **kwargs
) -> dict:
    curr_point = start
    min_error = None
    run_avg = np.zeros(np.size(start))
    curr_iter = 0
    curr_value = None
    n = X.shape[0]
    if L is not None:
        L_last = None
    if use_epoch:
        curr_epoch = 0

    if type(batch_size) is float and batch_size < 1:
        batch_size = n * batch_size

    batch_size = int(batch_size)

    while min_error is None or (curr_iter < max_iter and min_error >= tol):
        if use_epoch:
            idx = permutation(n)
            X_perm, y_perm = X[idx], y[idx]

            if L is not None:
                if curr_epoch == 0:
                    L_start = L(curr_point, X, y, **kwargs)
                    L_last = L_start
                elif curr_epoch % n_iter_no_change == 0 and curr_epoch != 0:
                    if np.abs(L_last - L_start) < tol:
                        learning_rate /= 5
                    L_last = L_start

            for batch_start in range(0, n, batch_size):
                batch_end = batch_start + batch_size
                batch_X, batch_y = (
                    X_perm[batch_start:batch_end],
                    y_perm[batch_start:batch_end],
                )
                curr_grad = L_grad(curr_point, batch_X, batch_y, **kwargs)

                run_avg = decay_rate * run_avg + (1 - decay_rate) * curr_grad**2

                curr_point -= learning_rate / np.sqrt(run_avg) * curr_grad
                curr_iter += 1

            W_error = norm(learning_rate * curr_grad)
            if L is not None:
                L_new = L(curr_point, X, y, **kwargs)
                L_error = np.abs(L_start - L_new)
                L_start = min(L_new, L_start)
                min_error = min(W_error, L_error)
            else:
                min_error = W_error

            curr_epoch += 1

        else:
            idx = choice(n, batch_size, replace=False)
            batch_X, batch_y = X[idx], y[idx]
            if L is not None:
                L_start = L(curr_point, X, y, **kwargs)

            curr_grad = L_grad(curr_point, batch_X, batch_y, **kwargs)
            run_avg = decay_rate * run_avg + (1 - decay_rate) * curr_grad**2

            curr_point -= learning_rate / np.sqrt(run_avg) * curr_grad

            W_error = norm(learning_rate * curr_grad)
            if L is not None:
                L_error = np.abs(L_start - L(curr_point, X, y, **kwargs))
                min_error = min(W_error, L_error)
            else:
                min_error = W_error

            curr_iter += 1

    if L is not None:
        curr_value = L(curr_point, X, y, **kwargs)

    return {
        "point": curr_point,
        "L_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
