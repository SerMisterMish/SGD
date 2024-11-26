import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma, digamma
from typing import Callable


def logit_inverse(x: NDArray, beta: NDArray) -> NDArray:
    t = np.exp(x.dot(beta))
    return t / (1 + t)


def logit_deriv(mu: NDArray) -> NDArray:
    return 1 / (mu - mu * mu)


def beta_log_likelihood(
    parameters: NDArray,
    X: NDArray,
    y: NDArray,
    link_inverse: Callable[[NDArray, NDArray], float],
    link_deriv=None,
) -> float:
    beta, phi = parameters[:-1], parameters[-1]
    mu = link_inverse(X, beta)

    prod = mu * phi
    return np.sum(
        np.log(gamma(phi))
        - np.log(gamma(prod))
        - np.log(gamma(phi - prod))
        + (prod - 1) * np.log(y)
        + (phi - prod - 1) * np.log(1 - y)
    )


def beta_inv_log_likelihood(
    parameters: NDArray,
    X: NDArray,
    y: NDArray,
    link_inverse: Callable[[NDArray, NDArray], float],
    link_deriv=None,
) -> float:
    return -beta_log_likelihood(parameters, X, y, link_inverse, link_deriv)


def beta_illh_grad(
    parameters: NDArray,
    X: NDArray,
    Y: NDArray,
    link_inverse: Callable[[NDArray, NDArray], NDArray],
    link_deriv: Callable[[NDArray], NDArray],
) -> NDArray:
    beta, phi = parameters[:-1], parameters[-1]
    mu_vec = link_inverse(X, beta)
    Y_star = np.log(Y / (1 - Y))
    mu_star = digamma(mu_vec * phi) - digamma((1 - mu_vec) * phi)
    T_mat = np.diag(1 / link_deriv(mu_vec))
    L_beta = phi * X.T.dot(T_mat).dot(Y_star - mu_star)
    L_phi = np.sum(
        mu_vec * (Y_star - mu_star)
        + np.log(1 - Y)
        - digamma((1 - mu_vec) * phi)
        + digamma(phi)
    )
    return -np.append(L_beta, L_phi)
