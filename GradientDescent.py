from numpy.linalg import norm
from numpy.typing import NDArray
from typing import Callable


def GradientDescent(
    start: NDArray,
    f: Callable,
    f_grad: Callable,
    learning_rate: float = 0.01,
    max_iter=1000,
    tol=1e-7,
    **kwargs
) -> dict:
    curr_point = start
    curr_value, prev_value = f(curr_point, **kwargs), None
    curr_grad = f_grad(curr_point, **kwargs)

    curr_iter = 0
    while curr_iter == 0 or (
        0 < curr_iter < max_iter
        and norm(curr_grad) >= tol
        and abs(curr_value - prev_value) >= tol
    ):
        temp_point = curr_point - learning_rate * curr_grad
        temp_value = f(temp_point, **kwargs)
        curr_point = temp_point
        prev_value, curr_value = curr_value, temp_value
        curr_grad = f_grad(curr_point, **kwargs)
        curr_iter += 1

    return {
        "point": curr_point,
        "f_value": curr_value,
        "grad_value": curr_grad,
        "iterations": curr_iter,
    }
