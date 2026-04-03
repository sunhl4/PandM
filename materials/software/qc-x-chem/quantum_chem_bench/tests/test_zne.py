"""Tests for ZNE error mitigation module."""

import pytest
import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from quantum_chem_bench.error_mitigation.zne import extrapolate_zne, ZNEWrapper


class TestExtrapolateZNE:
    def test_richardson_linear(self):
        """For linear noise, Richardson extrapolation should be exact."""
        exact = -1.0
        lambdas = [1.0, 3.0]
        evs = [exact + 0.1 * lam for lam in lambdas]
        result = extrapolate_zne(lambdas, evs, method="richardson")
        assert abs(result - exact) < 1e-10

    def test_richardson_quadratic(self):
        """3-point Richardson for quadratic noise."""
        exact = -1.5
        lambdas = [1.0, 3.0, 5.0]
        evs = [exact + 0.02 * lam + 0.001 * lam**2 for lam in lambdas]
        result = extrapolate_zne(lambdas, evs, method="richardson")
        assert abs(result - exact) < 1e-5

    def test_polynomial_extrapolation(self):
        exact = -2.0
        lambdas = [1.0, 2.0, 3.0, 4.0]
        evs = [exact + 0.05 * lam for lam in lambdas]
        result = extrapolate_zne(lambdas, evs, method="polynomial")
        assert abs(result - exact) < 1e-9

    def test_single_point_not_raises(self):
        """Single point extrapolation (no noise reduction, returns same value)."""
        lambdas = [1.0]
        evs = [-1.0]
        result = extrapolate_zne(lambdas, evs, method="richardson")
        assert abs(result - (-1.0)) < 1e-10

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown ZNE method"):
            extrapolate_zne([1, 3], [-1.0, -0.9], method="magic")


class TestZNEWrapper:
    def test_wrapper_calls_fn_at_each_scale(self):
        calls = []

        def mock_energy_fn(circuit, **kwargs):
            sf = getattr(circuit, "_scale", 1.0)
            calls.append(sf)
            return -1.0 + 0.1 * sf

        # Patch fold_gates to return a tagged circuit
        from unittest.mock import patch, MagicMock
        import quantum_chem_bench.error_mitigation.zne as zne_mod

        mock_circuit = MagicMock()

        def mock_fold(circuit, sf):
            c = MagicMock()
            c._scale = sf
            return c

        with patch.object(zne_mod, "fold_gates", side_effect=mock_fold):
            wrapper = ZNEWrapper(mock_energy_fn, scale_factors=[1.0, 3.0, 5.0])
            result = wrapper.run(mock_circuit)

        assert len(result["scale_factors"]) == 3
        assert "zero_noise_energy" in result
        # Richardson extrapolation of linear: exact = -1.0
        assert abs(result["zero_noise_energy"] - (-1.0)) < 1e-8

    def test_wrapper_stores_expectations(self):
        from unittest.mock import patch, MagicMock
        import quantum_chem_bench.error_mitigation.zne as zne_mod

        def mock_energy_fn(circuit, **kwargs):
            return -0.5

        with patch.object(zne_mod, "fold_gates", return_value=MagicMock()):
            wrapper = ZNEWrapper(mock_energy_fn, scale_factors=[1.0, 3.0])
            result = wrapper.run(MagicMock())

        assert len(result["expectations"]) == 2
        assert all(abs(e - (-0.5)) < 1e-10 for e in result["expectations"])
