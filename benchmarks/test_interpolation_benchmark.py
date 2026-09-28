"""Full-pipeline benchmarks for automatic jaxcapse interpolation."""

import jax
import numpy as np
import pytest

from jaxcapse import load_emulator
from tests.fixtures import *


@pytest.fixture
def interpolation_benchmark_data(mock_emulator_directory, sample_cosmological_params):
    np.save(
        mock_emulator_directory / "l.npy",
        np.linspace(2.0, 200.0, 100),
    )
    cubic = load_emulator(str(mock_emulator_directory))
    identity = load_emulator(str(mock_emulator_directory), interpolation="none")
    params = jax.device_put(sample_cosmological_params)

    cubic.get_Cl(params).block_until_ready()
    identity.get_Cl(params).block_until_ready()
    return cubic, identity, params


@pytest.mark.parametrize("method_name", ("identity", "cubic"))
def test_prediction_runtime(benchmark, interpolation_benchmark_data, method_name):
    cubic, identity, params = interpolation_benchmark_data
    emulator = cubic if method_name == "cubic" else identity

    def run():
        result = emulator.get_Cl(params)
        result.block_until_ready()
        return result

    benchmark.pedantic(run, rounds=30, iterations=1, warmup_rounds=5)
    benchmark.extra_info["training_knots"] = 100
    benchmark.extra_info["prediction_multipoles"] = len(emulator.get_ell_grid())
    benchmark.extra_info["jit"] = True
    benchmark.extra_info["device"] = str(jax.devices()[0])
