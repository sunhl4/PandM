"""
Zero Noise Extrapolation (ZNE) — error mitigation for near-term quantum circuits.

Reference: Li & Benjamin, PRX 7, 021050 (2017);
           Temme et al., PRL 119, 180509 (2017);
           Used in Arute et al., Science 369, 1084 (2020) (science.abd3880).

ZNE procedure:
    1. Execute the circuit at scale factors λ₁ < λ₂ < … < λₙ.
       Each scaled circuit has its noise amplified by inserting gate pairs
       (e.g. gate + gate†gate for λ=3 "folding").
    2. Measure ⟨H⟩_λ at each scale factor.
    3. Extrapolate to λ→0 (zero noise) using polynomial or Richardson
       extrapolation.

This module provides:
    - ``ZNEWrapper``     : wraps any VQE-style solver with ZNE post-processing.
    - ``fold_gates``     : circuit noise amplification via gate folding.
    - ``extrapolate_zne``: polynomial Richardson extrapolation.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Richardson / polynomial extrapolation
# ---------------------------------------------------------------------------

def extrapolate_zne(
    scale_factors: Sequence[float],
    expectations: Sequence[float],
    method: str = "richardson",
    order: int | None = None,
) -> float:
    """
    Extrapolate noisy expectation values to zero noise.

    Parameters
    ----------
    scale_factors : sequence of float
        Noise scale factors used (e.g. [1, 3, 5]).
    expectations : sequence of float
        Measured ⟨H⟩ values at the corresponding scale factors.
    method : str
        ``"richardson"`` (default) or ``"polynomial"``.
    order : int or None
        Polynomial degree for fitting. If None, uses len(scale_factors)-1.

    Returns
    -------
    float
        Extrapolated zero-noise expectation value.
    """
    lambdas = np.array(scale_factors, dtype=float)
    evs = np.array(expectations, dtype=float)
    n = len(lambdas)

    if order is None:
        order = n - 1

    if method == "richardson":
        # Richardson extrapolation: E₀ = Σᵢ cᵢ E(λᵢ)
        # where cᵢ = Πⱼ≠ᵢ λⱼ / (λⱼ - λᵢ)  for λ→0 extrapolation
        coeffs = np.zeros(n)
        for i in range(n):
            c = 1.0
            for j in range(n):
                if j != i:
                    c *= lambdas[j] / (lambdas[j] - lambdas[i])
            coeffs[i] = c
        return float(np.dot(coeffs, evs))

    elif method == "polynomial":
        # Polynomial fit + evaluate at λ=0
        poly = np.polyfit(lambdas, evs, deg=min(order, n - 1))
        return float(np.polyval(poly, 0.0))

    else:
        raise ValueError(f"Unknown ZNE method: '{method}'")


# ---------------------------------------------------------------------------
# Circuit noise amplification via gate folding
# ---------------------------------------------------------------------------

def fold_gates(circuit, scale_factor: float):
    """
    Amplify circuit noise by global gate folding.

    For integer scale factor k (must be odd): each gate G is replaced by
    G(G†G)^{(k-1)/2}, tripling the gate count for k=3, etc.

    Parameters
    ----------
    circuit : QuantumCircuit
        Original Qiskit circuit.
    scale_factor : float
        Noise scale factor (1 = original, 3 = triple noise, etc.).

    Returns
    -------
    QuantumCircuit
        Noise-amplified circuit.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise ImportError("qiskit required") from exc

    if abs(scale_factor - 1.0) < 1e-6:
        return circuit.copy()

    k = int(round(scale_factor))
    if k % 2 == 0:
        k += 1  # Must be odd for gate folding
        logger.warning("scale_factor rounded to odd integer k=%d for gate folding.", k)

    num_extra = (k - 1) // 2  # Number of G†G pairs to append per gate

    # Decompose to basis gates
    from qiskit.compiler import transpile
    basis = ["cx", "u3", "id", "x", "y", "z", "h", "s", "t", "rx", "ry", "rz"]
    try:
        decomposed = transpile(circuit, basis_gates=basis, optimization_level=0)
    except Exception:  # noqa: BLE001
        decomposed = circuit.copy()

    folded = QuantumCircuit(*decomposed.qregs, *decomposed.cregs)
    for instruction in decomposed.data:
        gate = instruction.operation
        qargs = instruction.qubits
        cargs = instruction.clbits

        folded.append(gate, qargs, cargs)
        for _ in range(num_extra):
            folded.append(gate.inverse(), qargs, cargs)
            folded.append(gate, qargs, cargs)

    return folded


# ---------------------------------------------------------------------------
# ZNE wrapper for arbitrary energy evaluation functions
# ---------------------------------------------------------------------------

class ZNEWrapper:
    """
    Apply ZNE to any function that maps a (possibly noisy) circuit to an energy.

    Parameters
    ----------
    energy_fn : callable
        ``energy_fn(circuit, **kwargs) → float``
        Evaluates ⟨H⟩ for a given circuit (possibly with noise).
    scale_factors : list of float
        Noise scale factors to use (default: [1, 3, 5]).
    extrapolation : str
        Extrapolation method: ``"richardson"`` or ``"polynomial"``.

    Example::

        def noisy_energy(circuit):
            # run with AerSimulator noise model
            ...

        zne = ZNEWrapper(noisy_energy, scale_factors=[1, 3, 5])
        zero_noise_energy = zne.run(ansatz_circuit)
    """

    def __init__(
        self,
        energy_fn: Callable,
        scale_factors: list[float] | None = None,
        extrapolation: str = "richardson",
    ) -> None:
        self.energy_fn = energy_fn
        self.scale_factors = scale_factors or [1.0, 3.0, 5.0]
        self.extrapolation = extrapolation

    def run(self, circuit, **kwargs) -> dict:
        """
        Run ZNE: evaluate at each scale factor, then extrapolate.

        Returns
        -------
        dict with keys:
            ``"zero_noise_energy"`` — extrapolated energy,
            ``"scale_factors"``    — scale factors used,
            ``"expectations"``     — measured energies at each scale.
        """
        expectations = []
        for sf in self.scale_factors:
            folded = fold_gates(circuit, sf)
            ev = self.energy_fn(folded, **kwargs)
            expectations.append(float(ev))
            logger.debug("ZNE scale=%.1f → E = %.8f", sf, ev)

        zero_noise = extrapolate_zne(
            self.scale_factors, expectations, method=self.extrapolation
        )
        logger.info("ZNE extrapolated zero-noise energy: %.10f", zero_noise)

        return {
            "zero_noise_energy": zero_noise,
            "scale_factors": list(self.scale_factors),
            "expectations": expectations,
        }
