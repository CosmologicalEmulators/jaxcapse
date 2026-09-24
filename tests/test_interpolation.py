"""Tests for transparent spline interpolation in jaxcapse emulators."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxace import CubicSplinePlan, inv_maximin, maximin
from jaxcapse import (
    IdentityInterpolation,
    SplinePlan,
    load_emulator,
    prepare_interpolation_method,
)
from jaxcapse.jaxcapse import _resolve_training_ell_grid
from tests.fixtures import *

jax.config.update("jax_enable_x64", True)

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def julia_reference():
    inputs = np.loadtxt(DATA_DIR / "spline_reference_inputs.txt")
    outputs = np.loadtxt(DATA_DIR / "spline_reference_outputs.txt")
    return inputs, outputs


def test_spline_plan_matches_saved_julia_reference(julia_reference):
    inputs, outputs = julia_reference
    training_grid = jnp.asarray(inputs[:, 0])
    values = jnp.asarray(inputs[:, 1:3])
    plan = SplinePlan(training_grid)

    # The high-level policy targets integer multipoles 0:5. The saved Julia
    # fixture uses spacing 0.05, so every twentieth row is the same target grid.
    expected = outputs[::20, 3:5]
    result = plan(values)

    assert isinstance(plan.Plan, CubicSplinePlan)
    assert plan.SourceAscending
    np.testing.assert_array_equal(plan.PredictionEllGrid, np.arange(0, 6))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_spline_plan_descending_matrix_and_jit(julia_reference):
    inputs, _ = julia_reference
    training_grid = jnp.asarray(inputs[:, 0])
    values = jnp.asarray(inputs[:, 1:3])
    ascending = SplinePlan(training_grid)
    descending = SplinePlan(training_grid[::-1])

    compiled = jax.jit(descending)
    result = compiled(values[::-1, :])
    result.block_until_ready()

    assert not descending.SourceAscending
    np.testing.assert_allclose(result, ascending(values), rtol=1e-12, atol=1e-12)


def test_automatic_interpolation_policy():
    dense_grid = np.arange(2, 102)
    sparse_grid = np.linspace(2, 200, 100)
    oversized_grid = np.linspace(2, 5000, 2049)

    assert isinstance(
        prepare_interpolation_method(dense_grid),
        IdentityInterpolation,
    )
    assert isinstance(prepare_interpolation_method(sparse_grid), SplinePlan)
    assert isinstance(
        prepare_interpolation_method(oversized_grid),
        IdentityInterpolation,
    )
    assert isinstance(
        prepare_interpolation_method(sparse_grid, interpolation="none"),
        IdentityInterpolation,
    )
    assert isinstance(
        prepare_interpolation_method(dense_grid, interpolation="cubic"),
        SplinePlan,
    )

    with pytest.raises(ValueError, match="at least two"):
        prepare_interpolation_method([2.0])
    with pytest.raises(ValueError, match="strictly monotonic"):
        prepare_interpolation_method([2.0, 5.0, 4.0])
    with pytest.raises(ValueError, match="max_spline_knots"):
        prepare_interpolation_method(sparse_grid, max_spline_knots=1)
    with pytest.raises(ValueError, match="interpolation must be"):
        prepare_interpolation_method(sparse_grid, interpolation="nonsense")


def test_tolerant_integer_endpoint_inference():
    np.testing.assert_array_equal(
        SplinePlan([2.0001, 5.0, 9.999]).PredictionEllGrid,
        np.arange(2, 11),
    )
    np.testing.assert_array_equal(
        SplinePlan([2.999, 5.0, 9.999]).PredictionEllGrid,
        np.arange(3, 11),
    )
    np.testing.assert_array_equal(
        SplinePlan([2.101, 5.0, 9.899]).PredictionEllGrid,
        np.arange(3, 10),
    )
    np.testing.assert_array_equal(
        SplinePlan([2.5, 5.0, 10.5]).PredictionEllGrid,
        np.arange(3, 11),
    )

    n_nodes = 512
    indices = np.arange(1, n_nodes + 1)
    roots = np.cos((2 * indices - 1) * np.pi / (2 * n_nodes))
    first_kind_grid = np.sort(4501.0 + 4499.0 * roots)
    np.testing.assert_array_equal(
        SplinePlan(first_kind_grid).PredictionEllGrid,
        np.arange(2, 9001),
    )

    with pytest.raises(ValueError, match="endpoint_tolerance"):
        SplinePlan([2.0, 5.0, 10.0], endpoint_tolerance=0.5)


def test_multipole_grid_must_match_network_output(mock_emulator_directory):
    grid = np.arange(2, 102)
    np.testing.assert_array_equal(_resolve_training_ell_grid(grid, 100), grid)

    for mismatched_grid, output_length in (
        (np.arange(0, 13), 4),
        (np.arange(0, 10051), 4999),
        (np.linspace(3, 20, 10), 5),
    ):
        with pytest.raises(ValueError, match="does not match"):
            _resolve_training_ell_grid(mismatched_grid, output_length)

    # This zero-origin grid previously loaded by silently assigning outputs
    # to ell=2:101, despite having 103 grid entries for 100 network outputs.
    np.save(mock_emulator_directory / "l.npy", np.arange(0, 103))
    with pytest.raises(ValueError, match="does not match"):
        load_emulator(str(mock_emulator_directory))


def test_dense_grid_identity_returns_original_array():
    method = prepare_interpolation_method(np.arange(2, 102))
    values = jnp.arange(100.0)
    assert method(values) is values


@pytest.fixture
def sparse_emulator(mock_emulator_directory):
    np.save(
        mock_emulator_directory / "l.npy",
        np.linspace(2.0, 200.0, 100),
    )
    return load_emulator(str(mock_emulator_directory))


def test_loader_stores_camel_case_grid_and_method_fields(sparse_emulator):
    emulator = sparse_emulator
    assert isinstance(emulator.InterpolationMethod, SplinePlan)
    assert isinstance(emulator.InterpolationMethod.Plan, CubicSplinePlan)
    np.testing.assert_allclose(
        emulator.TrainingEllGrid,
        np.linspace(2.0, 200.0, 100),
    )
    np.testing.assert_array_equal(emulator.PredictionEllGrid, np.arange(2, 201))
    np.testing.assert_array_equal(
        emulator.get_training_ell_grid(),
        emulator.TrainingEllGrid,
    )
    np.testing.assert_array_equal(
        emulator.get_ell_grid(),
        emulator.PredictionEllGrid,
    )


def test_get_cl_transparently_applies_stored_interpolation(
    sparse_emulator,
    sample_cosmological_params,
):
    emulator = sparse_emulator
    params = jnp.asarray(sample_cosmological_params)

    normalized_input = maximin(params, emulator.in_MinMax)
    normalized_output = emulator.emulator.run_emulator(normalized_input)
    output = inv_maximin(normalized_output, emulator.out_MinMax)
    processed = emulator.postprocessing(params, output)
    expected = emulator.InterpolationMethod(processed)
    result = emulator.get_Cl(params)

    assert result.shape == (199,)
    assert len(result) == len(emulator.get_ell_grid())
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_transparent_interpolation_batch_jit_and_ad(
    sparse_emulator,
    sample_cosmological_params,
):
    emulator = sparse_emulator
    params = jnp.asarray(sample_cosmological_params)
    second_params = params.at[0].set(params[0] + 0.01)

    first = emulator.get_Cl(params)
    second = emulator.get_Cl(second_params)
    first.block_until_ready()
    second.block_until_ready()
    assert not np.allclose(first, second)

    batch = jnp.stack((params, second_params))
    batch_result = emulator.get_Cl_batch(batch)
    assert batch_result.shape == (2, 199)
    np.testing.assert_allclose(batch_result[0], first, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(batch_result[1], second, rtol=1e-12, atol=1e-12)

    jacobian_forward = jax.jacfwd(emulator.get_Cl)(params)
    jacobian_reverse = jax.jacrev(emulator.get_Cl)(params)
    assert jacobian_forward.shape == (199, 6)
    assert jacobian_reverse.shape == (199, 6)
    np.testing.assert_allclose(
        jacobian_reverse,
        jacobian_forward,
        rtol=1e-9,
        atol=1e-9,
    )


def test_loader_interpolation_overrides(
    mock_emulator_directory,
    sample_cosmological_params,
):
    no_interpolation = load_emulator(
        str(mock_emulator_directory),
        interpolation="none",
    )
    forced_cubic = load_emulator(
        str(mock_emulator_directory),
        interpolation="cubic",
    )

    assert isinstance(no_interpolation.InterpolationMethod, IdentityInterpolation)
    assert isinstance(forced_cubic.InterpolationMethod, SplinePlan)
    assert no_interpolation.get_Cl(sample_cosmological_params).shape == (100,)
    assert forced_cubic.get_Cl(sample_cosmological_params).shape == (100,)
