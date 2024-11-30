from numpy.linalg import norm
from numpy.typing import NDArray
from typing import Callable


def GradientDescent(
    start: NDArray,
    f_grad: Callable,
    f: Callable | None = None,
    learning_rate: float = 0.01,
    max_iter=1000,
    tol=1e-4,
    **kwargs
) -> dict:
    curr_point = start
    curr_value = None
    curr_grad = f_grad(curr_point, **kwargs)

    curr_iter = 0
    while curr_iter == 0 or (
        curr_iter < max_iter and learning_rate * norm(curr_grad) >= tol
    ):
        curr_point = curr_point - learning_rate * curr_grad
        curr_grad = f_grad(curr_point, **kwargs)
        curr_iter += 1

    if f is not None:
        curr_value = f(curr_point, **kwargs)

    return {
        "point": curr_point,
        "f_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
