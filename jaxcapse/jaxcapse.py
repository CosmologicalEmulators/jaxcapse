from typing import Any, Callable, Dict, Optional
import numpy as np
import jax
import jax.numpy as jnp
import json
import importlib.util
import os
from functools import partial

# Import jaxace components (required dependency in pyproject.toml)
from jaxace import (
    CubicSplinePlan,
    init_emulator,
    FlaxEmulator,
    maximin,
    inv_maximin,
)

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


class IdentityInterpolation:
    """Return emulator outputs unchanged on their training grid."""

    def __init__(self, prediction_ell_grid):
        self.PredictionEllGrid = jnp.asarray(prediction_ell_grid)

    def __call__(self, values):
        return values


class SplinePlan:
    """Interpolate fixed training multipoles onto a dense integer grid."""

    def __init__(
        self,
        training_ell_grid,
        plan_type=CubicSplinePlan,
        endpoint_tolerance=0.1,
    ):
        training_grid = np.asarray(training_ell_grid)
        source_ascending = _source_grid_orientation(training_grid)
        ell_min, ell_max = training_grid.min(), training_grid.max()
        dense_min = _dense_endpoint(ell_min, "left", endpoint_tolerance)
        dense_max = _dense_endpoint(ell_max, "right", endpoint_tolerance)
        if dense_min > dense_max:
            raise ValueError("The inferred dense multipole grid is empty")

        prediction_grid = np.arange(dense_min, dense_max + 1)
        knots = training_grid if source_ascending else training_grid[::-1]

        self.Plan = plan_type(jnp.asarray(knots), jnp.asarray(prediction_grid))
        self.PredictionEllGrid = jnp.asarray(prediction_grid)
        self.SourceAscending = source_ascending

    def __call__(self, values):
        ordered_values = values if self.SourceAscending else values[::-1, ...]
        return self.Plan(ordered_values)


def _source_grid_orientation(training_ell_grid):
    if np.ndim(training_ell_grid) != 1:
        raise ValueError("The multipole grid must be one-dimensional")
    if len(training_ell_grid) < 2:
        raise ValueError("The multipole grid needs at least two points")
    differences = np.diff(training_ell_grid)
    if np.all(differences > 0):
        return True
    if np.all(differences < 0):
        return False
    raise ValueError("The multipole grid must be strictly monotonic")


def _is_dense_integer_grid(training_ell_grid, source_ascending):
    ordered_grid = (
        training_ell_grid if source_ascending else training_ell_grid[::-1]
    )
    ell_min, ell_max = ordered_grid.min(), ordered_grid.max()
    if not float(ell_min).is_integer() or not float(ell_max).is_integer():
        return False
    dense_grid = np.arange(int(ell_min), int(ell_max) + 1)
    return ordered_grid.shape == dense_grid.shape and np.array_equal(
        ordered_grid,
        dense_grid,
    )


def _dense_endpoint(value, side, tolerance):
    if not 0 <= tolerance < 0.5:
        raise ValueError("endpoint_tolerance must satisfy 0 <= tolerance < 0.5")
    nearest_integer = round(float(value))
    if abs(float(value) - nearest_integer) <= tolerance:
        return int(nearest_integer)
    if side == "left":
        return int(np.ceil(value))
    return int(np.floor(value))


def _resolve_training_ell_grid(training_ell_grid, output_length):
    training_grid = np.asarray(training_ell_grid)
    if len(training_grid) != output_length:
        raise ValueError(
            f"The multipole grid length ({len(training_grid)}) does not match "
            f"the emulator output length ({output_length})"
        )
    return training_grid


def prepare_interpolation_method(
    training_ell_grid,
    interpolation="auto",
    max_spline_knots=2048,
    endpoint_tolerance=0.1,
):
    """Choose identity or cubic interpolation for an emulator training grid."""
    if max_spline_knots < 2:
        raise ValueError("max_spline_knots must be at least two")
    if not 0 <= endpoint_tolerance < 0.5:
        raise ValueError("endpoint_tolerance must satisfy 0 <= tolerance < 0.5")

    training_grid = np.asarray(training_ell_grid)
    source_ascending = _source_grid_orientation(training_grid)

    if interpolation == "none":
        return IdentityInterpolation(training_grid)
    if interpolation == "cubic":
        return SplinePlan(
            training_grid,
            endpoint_tolerance=endpoint_tolerance,
        )
    if interpolation != "auto":
        raise ValueError("interpolation must be 'auto', 'none', or 'cubic'")

    if len(training_grid) > max_spline_knots or _is_dense_integer_grid(
        training_grid,
        source_ascending,
    ):
        return IdentityInterpolation(training_grid)
    return SplinePlan(
        training_grid,
        endpoint_tolerance=endpoint_tolerance,
    )


class MLP:
    """
    CAPSE MLP emulator using jaxace infrastructure.

    This class wraps a jaxace FlaxEmulator with CAPSE-specific functionality
    for CMB power spectrum computation.
    """

    def __init__(self,
                 emulator: FlaxEmulator,
                 in_MinMax: np.ndarray,
                 out_MinMax: np.ndarray,
                 postprocessing: Callable,
                 emulator_description: Dict[str, Any],
                 training_ell_grid: np.ndarray,
                 interpolation: str = "auto",
                 max_spline_knots: int = 2048,
                 endpoint_tolerance: float = 0.1,
                 InterpolationMethod: Optional[Callable] = None):
        """
        Initialize MLP with jaxace emulator and CAPSE-specific components.

        Args:
            emulator: jaxace FlaxEmulator instance
            in_MinMax: Input normalization parameters
            out_MinMax: Output normalization parameters
            postprocessing: Postprocessing function (must be JAX-compatible)
            emulator_description: Emulator metadata
            training_ell_grid: Multipoles used while training the emulator
            interpolation: Automatic, disabled, or forced cubic interpolation
            max_spline_knots: Largest source grid automatically interpolated
            endpoint_tolerance: Distance used to snap source bounds to integers
            InterpolationMethod: Optional preconstructed interpolation method
        """
        self.emulator = emulator
        self.in_MinMax = jnp.asarray(in_MinMax)  # Ensure JAX arrays for JIT
        self.out_MinMax = jnp.asarray(out_MinMax)  # Ensure JAX arrays for JIT
        self.postprocessing = postprocessing
        self.emulator_description = emulator_description
        self.TrainingEllGrid = jnp.asarray(training_ell_grid)
        if len(self.TrainingEllGrid) != self.out_MinMax.shape[0]:
            raise ValueError(
                "The training multipole grid and emulator output dimensions "
                "must match"
            )
        self.InterpolationMethod = (
            prepare_interpolation_method(
                training_ell_grid,
                interpolation=interpolation,
                max_spline_knots=max_spline_knots,
                endpoint_tolerance=endpoint_tolerance,
            )
            if InterpolationMethod is None
            else InterpolationMethod
        )
        self.PredictionEllGrid = self.InterpolationMethod.PredictionEllGrid

    @partial(jax.jit, static_argnums=(0,))
    def get_Cl(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        Compute CMB power spectrum Cl values with JIT compilation.

        Args:
            input_data: Cosmological parameters as JAX array

        Returns:
            Processed Cl values
        """
        # Normalize input
        norm_input = maximin(input_data, self.in_MinMax)

        # Run through neural network using jaxace emulator
        norm_output = self.emulator.run_emulator(norm_input)

        # Denormalize output
        output = inv_maximin(norm_output, self.out_MinMax)

        # Apply postprocessing (assumed to be JAX-compatible)
        processed_output = self.postprocessing(input_data, output)

        return self.InterpolationMethod(processed_output)

    def get_Cl_batch(self, input_batch: np.ndarray) -> np.ndarray:
        """
        Compute CMB power spectrum Cl values for a batch of inputs using vectorization.

        Args:
            input_batch: Array of cosmological parameters, shape (n_samples, n_params)

        Returns:
            Array of processed Cl values, shape (n_samples, n_cls)
        """
        # Convert to JAX array
        input_jax = jnp.asarray(input_batch)

        # Vectorize the entire get_Cl function (already JIT-compiled)
        vmap_get_Cl = jax.vmap(self.get_Cl)

        # Process all inputs at once
        return vmap_get_Cl(input_jax)

    def predict(self, input_data: jnp.ndarray) -> jnp.ndarray:
        """
        Alias for get_Cl to maintain compatibility with tests and examples.

        Args:
            input_data: Cosmological parameters as JAX array

        Returns:
            Processed Cl values
        """
        return self.get_Cl(input_data)

    def get_ell_grid(self) -> jnp.ndarray:
        """Return the multipole grid corresponding to ``get_Cl`` output."""
        return self.PredictionEllGrid

    def get_training_ell_grid(self) -> jnp.ndarray:
        """Return the original emulator training multipoles."""
        return self.TrainingEllGrid


def load_preprocessing(root_path: str, filename: str) -> callable:
    """
    Load postprocessing function from Python file.

    Args:
        root_path: Directory containing the postprocessing file
        filename: Name of the postprocessing file (without .py extension)

    Returns:
        The postprocessing function
    """
    spec = importlib.util.spec_from_file_location(
        filename,
        os.path.join(root_path, f"{filename}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.postprocessing


def load_emulator(
    folder_path: str,
    interpolation: str = "auto",
    max_spline_knots: int = 2048,
    endpoint_tolerance: float = 0.1,
) -> MLP:
    """
    Load a CAPSE emulator using jaxace infrastructure.

    Args:
        folder_path: Path to the emulator folder containing:
            - nn_setup.json: Neural network specification
            - weights.npy: Trained weights
            - inminmax.npy: Input normalization parameters
            - outminmax.npy: Output normalization parameters
            - l.npy: Multipoles used during training
            - postprocessing.py: Postprocessing function
        interpolation: ``"auto"`` (default), ``"none"``, or ``"cubic"``.
        max_spline_knots: Maximum source-grid size selected for automatic cubic
            interpolation. Defaults to 2048.
        endpoint_tolerance: Maximum distance used to snap source bounds to an
            integer. Defaults to 0.1; bounds farther away are moved inward.

    Returns:
        MLP instance ready for inference
    """
    # Ensure folder path ends with /
    if not folder_path.endswith('/'):
        folder_path += '/'

    # Load CAPSE-specific files
    in_MinMax = jnp.load(os.path.join(folder_path, "inminmax.npy"))
    out_MinMax = jnp.load(os.path.join(folder_path, "outminmax.npy"))

    # Load neural network configuration
    config_path = os.path.join(folder_path, 'nn_setup.json')
    with open(config_path, 'r') as f:
        nn_dict = json.load(f)

    # Load weights
    weights = jnp.load(os.path.join(folder_path, "weights.npy"))

    # Initialize jaxace emulator with the neural network
    # jaxace now uses row-major (C) order by default, compatible with Python-trained models
    jaxace_emulator = init_emulator(
        nn_dict=nn_dict,
        weight=weights,
        validate=True  # Enable validation for safety
    )

    # Load CAPSE-specific postprocessing
    postprocessing = load_preprocessing(folder_path, "postprocessing")

    # Extract emulator description
    emulator_description = nn_dict.get("emulator_description", {})
    training_ell_grid = _resolve_training_ell_grid(
        jnp.load(os.path.join(folder_path, "l.npy")),
        nn_dict["n_output_features"],
    )

    # Create MLP instance with jaxace backend
    # JIT compilation happens automatically via the @jax.jit decorator
    return MLP(
        emulator=jaxace_emulator,
        in_MinMax=in_MinMax,
        out_MinMax=out_MinMax,
        postprocessing=postprocessing,
        emulator_description=emulator_description,
        training_ell_grid=training_ell_grid,
        interpolation=interpolation,
        max_spline_knots=max_spline_knots,
        endpoint_tolerance=endpoint_tolerance,
    )
