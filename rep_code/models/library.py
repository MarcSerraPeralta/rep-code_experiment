from typing import Sequence, Iterator

from stim import CircuitInstruction

from surface_sim.models import (
    DecoherenceNoiseModel,
    NoiselessModel,
    CircuitNoiseModel,
    ExperimentalNoiseModel,
)


class DecoherenceNoiseModelExp(DecoherenceNoiseModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        name = "X_ECHO"
        if self._sym_noise:
            duration = 0.5 * self.gate_duration(name)

            yield from self.idle(qubits, duration)
            yield CircuitInstruction("X", targets=self.get_inds(qubits))
            yield from self.idle(qubits, duration)
        else:
            duration = self.gate_duration(name)

            yield CircuitInstruction("X", targets=self.get_inds(qubits))
            yield from self.idle(qubits, duration)


class IncNoiseModelExp(NoiselessModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("X", targets=self.get_inds(qubits))
        for qubit in qubits:
            prob = self.params("sq_error_prob", qubit)
            yield CircuitInstruction(
                "DEPOLARIZE1", targets=self.get_inds([qubit]), gate_args=[prob]
            )


class ExperimentalNoiseModelExp(ExperimentalNoiseModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        duration = 0.5 * (self.gate_duration("X_ECHO") - self.gate_duration("X"))
        yield from self.idle(qubits, duration)
        yield from self.x_gate(qubits)
        yield from self.idle(qubits, duration)


class NoiselessModelExp(NoiselessModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("X", self.get_inds(qubits))


class CircuitNoiseModelExp(CircuitNoiseModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield from self.x_gate(qubits)
