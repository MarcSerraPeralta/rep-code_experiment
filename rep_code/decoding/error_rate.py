from typing import List

import numpy as np
import lmfit
import matplotlib.pyplot as plt
from uncertainties import ufloat


def error_prob_decay(
    r: np.ndarray,
    error_rate: float,
    r0: float,
) -> np.ndarray:
    return 0.5 - 0.5 * (1 - 2 * error_rate) ** (r - r0)


def fidelity(error_prob: np.ndarray) -> np.ndarray:
    return 1 - error_prob


def lmfit_par_to_ufloat(param: lmfit.parameter.Parameter) -> ufloat:
    """
    Safe conversion of an :class:`lmfit.parameter.Parameter` to
    :code:`uncertainties.ufloat(value, std_dev)`.
    """

    value = param.value
    stderr = np.nan if param.stderr is None else param.stderr

    return ufloat(value, stderr)


def get_error_rate(
    rounds: np.ndarray,
    log_errors: List[np.ndarray],
    distance: int,
    return_r0: bool = False,
) -> ufloat:
    """

    Parameters
    ----------
    log_errors: np.ndarray(Nrounds, Nshots)
    """
    error_prob = np.array([np.average(x) for x in log_errors])
    rounds = np.array(rounds)

    # fit only from distance
    error_prob = error_prob[rounds >= distance]
    rounds = rounds[rounds >= distance]

    guess = LogicalErrorProb().guess(error_prob, rounds)
    fit = LogicalErrorProb().fit(error_prob, guess, r=rounds)

    error_rate = lmfit_par_to_ufloat(fit.params["error_rate"])
    r0 = lmfit_par_to_ufloat(fit.params["r0"])

    if return_r0:
        return error_rate, r0

    return error_rate


class LogicalErrorProb(lmfit.model.Model):
    """
    lmfit model with a guess for a logical fidelity decay.
    """

    def __init__(self, fixed_t0=False):
        super().__init__(error_prob_decay)
        self.fixed_t0 = fixed_t0

        # configure constraints that are independent from the data to be fitted
        self.set_param_hint("error_rate", value=0.05, min=0, max=1, vary=True)
        if self.fixed_t0:
            self.set_param_hint("r0", value=0, vary=False)
        else:
            self.set_param_hint("r0", value=0, min=0, vary=True)

    def guess(
        self, data: np.ndarray, x: np.ndarray, **kws
    ) -> lmfit.parameter.Parameters:
        # to ensure they are np.ndarrays
        x, data = np.array(x), np.array(data)
        # guess parameters based on the data
        deriv_data = (data[1:] - data[:-1]) / (x[1:] - x[:-1])
        data_averaged = 0.5 * (data[1:] + data[:-1])
        error_rate_guess = 0.5 * (
            1 - np.exp(np.average(deriv_data / (data_averaged - 0.5)))
        )

        self.set_param_hint("error_rate", value=error_rate_guess)
        if not self.fixed_t0:
            self.set_param_hint("r0", value=0.01)
        params = self.make_params()

        return lmfit.models.update_param_vals(params, self.prefix, **kws)


def plot_fidelity_exp(
    ax: plt.Axes, rounds: np.ndarray, log_errors: np.ndarray, **kargs_errorbar
) -> plt.Axes:
    """

    Parameters
    ----------
    log_errors: np.ndarray(Nrounds, Nshots)
    """
    log_fid = np.array([np.average(fidelity(x)) for x in log_errors])
    log_fid_std = np.array([np.std(fidelity(x)) / np.sqrt(len(x)) for x in log_errors])

    ax.errorbar(rounds, log_fid, yerr=log_fid_std, **kargs_errorbar)

    ax.set_xlabel("QEC round, $r$")
    ax.set_ylabel("Logical fidelity, $F_L$")
    ax.set_xlim(xmin=0)
    ax.set_ylim(0.5, 1)

    return ax


def plot_fidelity_fit(
    ax: plt.Axes,
    rounds: np.ndarray,
    error_rate: float,
    r0: float,
    distance: int,
    **kargs_plot,
) -> plt.Axes:
    rounds_theo = np.linspace(np.min(rounds), np.max(rounds), 1_000)
    rounds_theo = rounds_theo[rounds_theo >= distance]
    log_fid = fidelity(error_prob_decay(rounds_theo, error_rate, r0))

    ax.plot(rounds_theo, log_fid, **kargs_plot)

    ax.set_xlabel("QEC round, $r$")
    ax.set_ylabel("Logical fidelity, $F_L$")
    ax.set_xlim(xmin=0)
    ax.set_ylim(0.5, 1)

    return ax


def plot_error_prob_exp(
    ax: plt.Axes, rounds: np.ndarray, log_errors: np.ndarray, **kargs_errorbar
) -> plt.Axes:
    """

    Parameters
    ----------
    log_errors: np.ndarray(Nrounds, Nshots)
    """
    error_prob = np.array([np.average(x) for x in log_errors])
    error_prob_std = np.array([np.std(x) / np.sqrt(len(x)) for x in log_errors])

    ax.errorbar(rounds, 1 - 2 * error_prob, yerr=error_prob_std, **kargs_errorbar)

    ax.set_xlabel("QEC round, $r$")
    ax.set_ylabel("Logical error probability, $1 - 2p_L$")
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=1)
    ax.set_yscale("log")

    return ax


def plot_error_prob_fit(
    ax: plt.Axes,
    rounds: np.ndarray,
    error_rate: float,
    r0: float,
    distance: int,
    **kargs_plot,
) -> plt.Axes:
    rounds_theo = np.linspace(np.min(rounds), np.max(rounds), 1_000)
    rounds_theo = rounds_theo[rounds_theo >= distance]
    error_prob = error_prob_decay(rounds_theo, error_rate, r0)

    ax.plot(rounds_theo, 1 - 2 * error_prob, **kargs_plot)

    ax.set_xlabel("QEC round, $r$")
    ax.set_ylabel("Logical error probability, $1 - 2p_L$")
    ax.set_xlim(xmin=0)
    ax.set_ylim(ymax=1)
    ax.set_yscale("log")

    return ax
