from surface_sim import Model

from rep_code.layout.rep_code_layout import get_rep_code_layout
from rep_code.circuits.repetition_code import memory_experiment


def test_repetition_code_circuit():
    layout = get_rep_code_layout(
        [
            "D1",
            "X1",
            "D2",
            "X2",
            "D3",
            "Z2",
            "D6",
            "Z4",
            "D5",
            "Z1",
            "D4",
            "Z2",
            "D7",
            "X3",
            "D8",
            "X4",
            "D9",
        ]
    )
    layout_to_ind = {q: layout.get_inds([q])[0] for q in layout.get_qubits()}
    model = Model(layout_to_ind)

    circuit = memory_experiment(
        model,
        layout,
        num_rounds=4,
        data_init=[
            0,
        ]
        * len(layout.get_qubits(role="data")),
    )

    circuit.diagram(type="detslice-with-ops-svg")

    return
