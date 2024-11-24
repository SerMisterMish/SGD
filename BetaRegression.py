import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma, digamma
from typing import Callable


def logit_inverse(x, beta) -> float:
    t = np.exp(np.inner(x, beta))
    return t / (1 + t)


def logit_deriv(mu) -> float:
    return 1 / (mu - mu * mu)


def beta_log_likelyhood_single(
    x: NDArray,
    y: float,
    beta: NDArray,
    phi: float,
    link_inverse: Callable[[NDArray, NDArray], float],
    mu: float | None = None,
) -> float:
    if mu is None:
        mu = link_inverse(x, beta)

    prod = mu * phi
    return (
        np.log(gamma(phi))
        - np.log(gamma(prod))
        - np.log(gamma(phi - prod))
        + (prod - 1) * np.log(y)
        + (phi - prod - 1) * np.log(1 - y)
    )


def beta_inv_log_likelyhood(
    parameters: NDArray,
    X: NDArray,
    Y: NDArray,
    link_inverse: Callable[[NDArray, NDArray], float],
    link_deriv=None,
) -> float | np.floating:
    beta, phi = parameters[:-1], parameters[-1]
    return -np.mean(
        [
            beta_log_likelyhood_single(x, y, beta, phi, link_inverse)
            for (x, y) in zip(X, Y)
        ]
    )


def beta_illh_grad(
    parameters: NDArray,
    X: NDArray,
    Y: NDArray,
    link_inverse: Callable[[NDArray, NDArray], float],
    link_deriv: Callable[[float], float],
) -> NDArray:
    beta, phi = parameters[:-1], parameters[-1]
    mu_vec = np.array([link_inverse(x, beta) for x in X])
    Y_star = np.array([np.log(y / (1 - y)) for y in Y])
    mu_star = np.array([digamma(mu * phi) - digamma((1 - mu) * phi) for mu in mu_vec])
    T_mat = np.diag([1 / link_deriv(mu) for mu in mu_vec])
    L_beta = phi * X.T.dot(T_mat).dot(Y_star - mu_star) / X.shape[0]
    L_phi = np.mean(
        [
            mu * (y_s - mu_s) + np.log(1 - y) - digamma((1 - mu) * phi) + digamma(phi)
            for (mu, y, mu_s, y_s) in zip(mu_vec, Y, mu_star, Y_star)
        ]
    )
    return -np.append(L_beta, L_phi)
