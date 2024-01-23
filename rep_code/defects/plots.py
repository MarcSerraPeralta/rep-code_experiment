import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def plot_average_defect_rate(
    ax: plt.Axes, defects: xr.DataArray, final_defects: xr.DataArray, **kargs_plot
) -> plt.Axes:
    """
    Plots defect rates corresponding to the average of all ancillas
    """

    # arguments for plotting
    kargs_plot_ = dict(label="average", linestyle="-", color="black", marker=".")
    kargs_plot_.update(kargs_plot)

    qec_round = defects.qec_round.values
    defects_rate = defects.mean(dim=["shot", "anc_qubit"]).values
    final_defects_rate = final_defects.mean(dim=["shot", "anc_qubit"]).values

    # append final defects
    qec_round = np.concatenate([qec_round, (qec_round[-1] + 1).reshape(1)])
    defects_rate = np.concatenate([defects_rate, [final_defects_rate]])

    ax.plot(qec_round, defects_rate, **kargs_plot_)

    ax.set_xlabel("QEC round, $r$")
    ax.set_ylabel("Defect rate")
    ax.set_xlim(0, qec_round[-1] + 1)

    return ax


def plot_defect_rates(
    ax: plt.Axes, defects: xr.DataArray, final_defects: xr.DataArray, **kargs_plot
) -> plt.Axes:
    """
    Plots defect rates for each ancilla
    """

    # arguments for plotting
    kargs_plot_ = dict(linestyle="-", color="gray", marker=".")
    kargs_plot_.update(kargs_plot)

    # extra round for final defects
    qec_round = defects.qec_round.values
    qec_round = np.concatenate([qec_round, (qec_round[-1] + 1).reshape(1)])

    for qubit in defects.anc_qubit:
        defects_rate = defects.sel(anc_qubit=qubit).mean(dim=["shot"]).values
        final_defects_rate = (
            final_defects.sel(anc_qubit=qubit).mean(dim=["shot"]).values
        )
        defects_rate = np.concatenate([defects_rate, [final_defects_rate]])

        kargs_plot_["label"] = str(qubit.values)
        ax.plot(qec_round, defects_rate, **kargs_plot_)

    ax.set_xlabel("QEC round, $r$")
    ax.set_ylabel("Defect rate")
    ax.set_xlim(0, qec_round[-1] + 1)
    ax.legend(loc="lower center")

    return ax
