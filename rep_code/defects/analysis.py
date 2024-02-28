from typing import Tuple

import numpy as np
from scipy.special import comb, factorial
from scipy.optimize import curve_fit


def binomial_dist(k: np.ndarray, n: float, p: float) -> float:
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def fit_binomial(k: np.ndarray, pdf: np.ndarray, n: float):
    bounds = ((0,), (1,))
    p0 = (np.sum(k * pdf) / n,)
    binomial_funct = lambda x, p: binomial_dist(x, n, p)
    popt, perr = curve_fit(
        binomial_funct,
        k,
        pdf,
        bounds=bounds,
        p0=p0,
    )
    return popt


def gaussian_dist(k: np.ndarray, mu: float, sigma: float) -> float:
    return np.exp(-0.5 * ((k - mu) / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)


def fit_gaussian(k: np.ndarray, pdf: np.ndarray):
    bounds = ((0, 1e-15), (np.max(k), np.max(k)))
    mu_guess = np.sum(k * pdf)
    p0 = (mu_guess, np.sum(pdf * (k - mu_guess) ** 2))
    popt, perr = curve_fit(
        gaussian_dist,
        k,
        pdf,
        bounds=bounds,
        p0=p0,
    )
    return popt


def poisson_dist(k: np.ndarray, l: float) -> float:
    return np.power(l, k) * np.exp(-l) / factorial(k)


def fit_poisson(k: np.ndarray, pdf: np.ndarray):
    bounds = ((0,), (np.max(k),))
    p0 = (np.sum(k * pdf),)
    popt, perr = curve_fit(
        poisson_dist,
        k,
        pdf,
        bounds=bounds,
        p0=p0,
    )
    return popt


def get_three_state_probs(
    classifier,
    iq_data: np.ndarray,
    prob_tol: float = 1e-3,
) -> Tuple[float, float, float]:
    """Estimates the probability of state 0, 1, 2 given
    the measured IQ values iteratively:
    1) predict IQ values and compute p_0, p_1, p_2
    2) update P(j|IQ) using p_j and Bayes' theorem

    Parameters
    ----------
    classifier
        Three-state classifier that has the attribute "predict(iq_data, p_0, p_1)"
    iq_data: np.array(..., 2)
        IQ data shots for which to compute p_0, p_1, p_2
    prob_tol
        Probability tolerance for the iterative calculation.
        The break condition is |p_i(previous) - p_i(new)| <= prob_tol.

    Returns
    -------
    p_0
        Probability that qubit is in state 0 in the given data
    p_1
        Probability that qubit is in state 1 in the given data
    p_2
        Probability that qubit is in state 2 in the given data
    """
    p_previous = np.ones(3) / 3
    p_new = np.zeros(3)
    num_total = np.prod(iq_data.shape[:-1])  # "iq" dimension

    while True:
        hard_outcomes = classifier.predict(iq_data, p0=p_previous[0], p1=p_previous[1])

        for k in range(3):
            p_new[k] = np.sum(hard_outcomes == k) / num_total

        if (np.abs(p_new - p_previous) <= prob_tol).all():
            break

        p_previous = p_new

    return p_new
