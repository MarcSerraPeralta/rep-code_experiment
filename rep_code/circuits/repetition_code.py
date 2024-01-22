from typing import Optional, Dict
from itertools import chain, compress

import networkx as nx
from stim import Circuit, CircuitInstruction, target_rec

from qec_util import Layout
from surface_sim import Model


def memory_experiment(
    model: Model,
    layout: Layout,
    num_rounds: int,
    data_init: Dict[str, bool],
    basis: str = "Z",
) -> Circuit:
    if not isinstance(num_rounds, int):
        raise ValueError(f"num_rounds expected as int, got {type(num_rounds)} instead.")
    if num_rounds <= 0:
        raise ValueError("num_rounds needs to be a positive integer")
    if basis == "Z":
        rot_basis = False
    elif basis == "X":
        rot_basis = True
    else:
        raise ValueError(f"'basis' must be 'Z' or 'X', but {basis} was given")
    if not isinstance(data_init, dict):
        raise ValueError(f"data_init expected as dict, got {type(num_rounds)} instead.")

    init_circ = init_qubits(model, layout, data_init=data_init, rot_basis=rot_basis)
    meas_circuit = log_meas(model, layout)
    first_qec_circ = qec_round(model, layout, meas_comparison=False)

    if num_rounds > 2:
        qec_circ = qec_round(model, layout)
        experiment = (
            init_circ + first_qec_circ * 2 + qec_circ * (num_rounds - 2) + meas_circuit
        )
    elif num_rounds == 2:
        experiment = (
            init_circ + first_qec_circ * 2 + log_meas(model, layout, comp_rounds=2)
        )
    elif num_rounds == 1:
        experiment = init_circ + first_qec_circ + log_meas(model, layout, comp_rounds=1)

    return experiment


def qec_round(model: Model, layout: Layout, meas_comparison: bool = True) -> Circuit:
    """
    Returns stim circuit corresponding to a QEC cycle from the repetition
    code run in the DiCarlo lab of the given model.

    Params
    -------
    model
        Noise model containing the noisy gates
    layout
        Layout of the device
    meas_comparison
        If True, the detector is set to the measurement of the ancilla
        instead of to the comparison of consecutive syndromes.

    Returns
    -------
    circuit
        Stim circuit containing the noisy operations of the QEc cycle
    """
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)
    qubit_coords = get_1d_coords(layout)

    int_order = layout.interaction_order

    circuit = Circuit()

    # step 1: hadamards
    rot_qubits = set(layout.get_qubits(role="anc", group=1))
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(rot_qubits)
    duration = model.gate_duration("H")
    for instruction in model.idle(idle_qubits, duration):
        circuit.append(instruction)

    for instruction in model.tick():
        circuit.append(instruction)

    # step 2 and 3: cz dance
    for group, order in int_order[:2]:
        cz_anc_qubits = layout.get_qubits(role="anc", group=group)
        int_qubits = []
        for q in cz_anc_qubits:
            direction = layout.param(qubit=q, param="order")[order]
            int_pairs = layout.get_neighbors(q, direction=direction, as_pairs=True)
            int_qubits += list(chain.from_iterable(int_pairs))

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

        idle_qubits = qubits - set(int_qubits)
        duration = model.gate_duration("CZ")
        for instruction in model.idle(idle_qubits, duration):
            circuit.append(instruction)

        for instruction in model.tick():
            circuit.append(instruction)

    # step 4: hadamards
    rot_qubits = anc_qubits
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(rot_qubits)
    duration = model.gate_duration("H")
    for instruction in model.idle(idle_qubits, duration):
        circuit.append(instruction)

    for instruction in model.tick():
        circuit.append(instruction)

    # step 5 and 6: cz dance
    for group, order in int_order[2:]:
        cz_anc_qubits = layout.get_qubits(role="anc", group=group)
        int_qubits = []
        for q in cz_anc_qubits:
            direction = layout.param(qubit=q, param="order")[order]
            int_pairs = layout.get_neighbors(q, direction=direction, as_pairs=True)
            int_qubits += list(chain.from_iterable(int_pairs))

        for instruction in model.cphase(int_qubits):
            circuit.append(instruction)

        idle_qubits = qubits - set(int_qubits)
        duration = model.gate_duration("CZ")
        for instruction in model.idle(idle_qubits, duration):
            circuit.append(instruction)

        for instruction in model.tick():
            circuit.append(instruction)

    # step 7: hadamards
    rot_qubits = set(layout.get_qubits(role="anc", group=2))
    for instruction in model.hadamard(rot_qubits):
        circuit.append(instruction)

    idle_qubits = qubits - set(rot_qubits)
    duration = model.gate_duration("H")
    for instruction in model.idle(idle_qubits, duration):
        circuit.append(instruction)

    for instruction in model.tick():
        circuit.append(instruction)

    # step 8: measurement with DD
    for instruction in model.measure(anc_qubits):
        circuit.append(instruction)

    for instruction in model.x_echo(data_qubits):
        circuit.append(instruction)

    # add detectors ordered as in the measurements
    num_anc = len(anc_qubits)
    if meas_comparison:
        det_targets = []
        for ind in range(num_anc):
            # If no meas_reset, then d[n] = m[n] ^ m[n-2]
            target_inds = [ind - (2 + 1) * num_anc, ind - num_anc]
            targets = [target_rec(ind) for ind in target_inds]
            det_targets.append(targets)
    else:
        det_targets = [[target_rec(ind - num_anc)] for ind in range(num_anc)]

    for anc_qubit, targets in zip(anc_qubits, det_targets):
        circuit.append("DETECTOR", targets=targets, arg=(qubit_coords[anc_qubit], 0))
    circuit.append("SHIFT_COORDS", arg=(0, 1))

    for instruction in model.tick():
        circuit.append(instruction)

    return circuit


def log_meas(model: Model, layout: Layout, comp_rounds: Optional[int] = 2) -> Circuit:
    """
    Returns stim circuit corresponding to a logicual measurement from the repetition
    code run in the DiCarlo lab of the given model.

    Params
    -------
    model
        Noise model containing the noisy gates
    layout
        Layout of the device
    comp_rounds
        Number of previous runs to use when defining the detectors.
        By default is 2 because without measurement-reset in ancillas
        then d_final = projected ^ m[N] ^ m[N-1]. The only edge case is when
        the circuit only has 1 QEC cycle.

    Returns
    -------
    circuit
        Stim circuit for the logical measurement
    """
    data_qubits = layout.get_qubits(role="data")
    anc_qubits = layout.get_qubits(role="anc")
    qubits = set(data_qubits + anc_qubits)
    qubit_coords = get_1d_coords(layout)

    circuit = Circuit()

    for instruction in model.measure(data_qubits):
        circuit.append(instruction)

    duration = model.gate_duration("M")
    for instruction in model.idle(anc_qubits, duration):
        circuit.append(instruction)

    num_data, num_anc = len(data_qubits), len(anc_qubits)
    for anc_qubit in anc_qubits:
        neighbors = layout.get_neighbors(anc_qubit)
        neighbor_inds = layout.get_inds(neighbors)
        targets = [target_rec(ind - num_data) for ind in neighbor_inds]

        anc_ind = anc_qubits.index(anc_qubit)
        for round_ind in range(1, comp_rounds + 1):
            # if no reset after ancilla measurements, then
            # d_final = projected ^ m[N] ^ m[N-1]
            target = target_rec(anc_ind - num_data - round_ind * num_anc)
            targets.append(target)
        circuit.append("DETECTOR", targets=targets, arg=(qubit_coords[anc_qubit], 0))
    circuit.append("SHIFT_COORDS", arg=(0, 1))

    targets = [target_rec(ind) for ind in range(-num_data, 0)]
    circuit.append("OBSERVABLE_INCLUDE", targets, 0)

    for instruction in model.tick():
        circuit.append(instruction)

    return circuit


def init_qubits(
    model: Model, layout: Layout, data_init: Dict[str, bool], rot_basis: bool = False
) -> Circuit:
    """
    Returns stim circuit corresponding to initialize the qubits from the repetition
    code run in the DiCarlo lab of the given model.

    Params
    -------
    model
        Noise model containing the noisy gates
    layout
        Layout of the device
    data_init
        Bitstring to initialize the data qubits
    rot_basis
        If True, initializes data qubits in X basis.
        By default, initializes data qubits in Z basis.

    Returns
    -------
    circuit
        Stim circuit for the logical measurement
    """
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    qubit_coords = {node: attr["coords"] for node, attr in layout.graph.nodes.items()}
    qubits = set(data_qubits + anc_qubits)

    circuit = Circuit()

    # add coordinates to circuit
    for qubit, coords in qubit_coords.items():
        coords = (coords[0], -coords[1])  # stim plots x, -y
        circuit.append(
            CircuitInstruction("QUBIT_COORDS", layout.get_inds([qubit]), coords)
        )

    # step 1: reset
    for instruction in model.reset(qubits):
        circuit.append(instruction)

    for instruction in model.tick():
        circuit.append(instruction)

    # step 2: X gates on bitstring data qubits
    # Note: compress outputs an iterable. If it is not converted
    # into a list, then the "if" statement processes the iterable
    # and the model.x_gate is called with an empty iterable.
    exc_qubits = list(compress(data_init.keys(), data_init.values()))
    if exc_qubits:
        for instruction in model.x_gate(exc_qubits):
            circuit.append(instruction)

    idle_qubits = qubits - set(exc_qubits)
    duration = model.gate_duration("X")
    for instruction in model.idle(idle_qubits, duration):
        circuit.append(instruction)

    for instruction in model.tick():
        circuit.append(instruction)

    # step 3(optional): hadamard gates for rotated basis
    if rot_basis:
        rot_qubits = data_qubits
        for instruction in model.hadamard(rot_qubits):
            circuit.append(instruction)

        idle_qubits = qubits - set(rot_qubits)
        duration = model.gate_duration("H")
        for instruction in model.idle(idle_qubits, duration):
            circuit.append(instruction)

        for instruction in model.tick():
            circuit.append(instruction)

    return circuit


def get_1d_coords(layout: Layout) -> Dict[str, int]:
    anc_qubits = layout.get_qubits(role="anc")
    data_qubits = layout.get_qubits(role="data")
    linear_graph = nx.Graph()
    for anc_qubit in anc_qubits:
        directions = layout.graph.nodes("order")[anc_qubit].values()
        for direction in directions:
            data_qubit = layout.get_neighbors(anc_qubit, direction=direction)[0]
            linear_graph.add_edge(anc_qubit, data_qubit)

    nodes_with_one_edge = [
        node for node, degree in linear_graph.degree() if degree == 1
    ]
    initial_qubit = sorted(nodes_with_one_edge)[0]
    qubit_line = sorted(
        layout.get_qubits(),
        key=lambda x: nx.shortest_path_length(
            linear_graph, source=initial_qubit, target=x
        ),
    )
    return {q: c for q, c in zip(qubit_line, range(len(layout.get_qubits())))}
