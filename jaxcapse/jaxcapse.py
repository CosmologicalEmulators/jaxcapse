from typing import Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
import json
import importlib.util
import os
from functools import partial

# Import jaxace components (required dependency in pyproject.toml)
from jaxace import (
    init_emulator,
    FlaxEmulator,
    maximin,
    inv_maximin
)

# Configure JAX for 64-bit precision
jax.config.update("jax_enable_x64", True)


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
                 postprocessing: callable,
                 emulator_description: Dict[str, Any]):
        """
        Initialize MLP with jaxace emulator and CAPSE-specific components.

        Args:
            emulator: jaxace FlaxEmulator instance
            in_MinMax: Input normalization parameters
            out_MinMax: Output normalization parameters
            postprocessing: Postprocessing function (must be JAX-compatible)
            emulator_description: Emulator metadata
        """
        self.emulator = emulator
        self.in_MinMax = jnp.asarray(in_MinMax)  # Ensure JAX arrays for JIT
        self.out_MinMax = jnp.asarray(out_MinMax)  # Ensure JAX arrays for JIT
        self.postprocessing = postprocessing
        self.emulator_description = emulator_description

        # For backward compatibility
        self.NN_params = emulator.parameters
        self.features = None  # Not directly accessible from FlaxEmulator
        self.activations = None  # Not directly accessible from FlaxEmulator

    def maximin_input(self, input_data: np.ndarray) -> np.ndarray:
        """Normalize input using min-max scaling."""
        return maximin(input_data, self.in_MinMax)

    def inv_maximin_output(self, output: np.ndarray) -> np.ndarray:
        """Denormalize output using inverse min-max scaling."""
        return inv_maximin(output, self.out_MinMax)

    def apply(self, params, x):
        """For backward compatibility - use emulator directly."""
        return self.emulator.model.apply(params, x)

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

        return processed_output

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


def load_emulator(folder_path: str) -> MLP:
    """
    Load a CAPSE emulator using jaxace infrastructure.

    Args:
        folder_path: Path to the emulator folder containing:
            - nn_setup.json: Neural network specification
            - weights.npy: Trained weights
            - inminmax.npy: Input normalization parameters
            - outminmax.npy: Output normalization parameters
            - postprocessing.py: Postprocessing function

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

    # Create MLP instance with jaxace backend
    # JIT compilation happens automatically via the @jax.jit decorator
    return MLP(
        emulator=jaxace_emulator,
        in_MinMax=in_MinMax,
        out_MinMax=out_MinMax,
        postprocessing=postprocessing,
        emulator_description=emulator_description
    )
