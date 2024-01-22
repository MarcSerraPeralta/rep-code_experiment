from typing import Sequence, Iterator

from stim import CircuitInstruction

from surface_sim.models import DecoherenceNoiseModel, NoiselessModel, CircuitNoiseModel


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


class NoiselessModelExp(NoiselessModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        yield CircuitInstruction("X", self.get_inds(qubits))


class CircuitNoiseModelExp(CircuitNoiseModel):
    def x_echo(self, qubits: Sequence[str]) -> Iterator[CircuitInstruction]:
        inds = self.get_inds(qubits)
        yield CircuitInstruction("X", inds)

        for qubit, ind in zip(qubits, inds):
            prob = self.param("sq_error_prob", qubit)
            yield CircuitInstruction("DEPOLARIZE1", [ind], [prob])
