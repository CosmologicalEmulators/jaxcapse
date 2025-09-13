"""
Shared test fixtures and utilities for jaxcapse tests.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import tempfile
import json
from pathlib import Path
import os


@pytest.fixture
def minimal_nn_dict():
    """Minimal valid neural network configuration for testing."""
    return {
        "n_input_features": 6,
        "n_output_features": 100,
        "n_hidden_layers": 2,
        "layers": {
            "layer_1": {"n_neurons": 32, "activation_function": "tanh"},
            "layer_2": {"n_neurons": 16, "activation_function": "relu"}
        },
        "emulator_description": {
            "author": "Test Suite",
            "author_email": "test@example.com",
            "parameters": "p1, p2, p3, p4, p5, p6",
            "miscellanea": "Test emulator for unit testing"
        }
    }


@pytest.fixture
def sample_weights(minimal_nn_dict):
    """Generate random weights matching the minimal network architecture."""
    total_size = 0
    
    # Input to first hidden layer: 6 * 32 + 32
    n_in = minimal_nn_dict["n_input_features"]
    n_out = minimal_nn_dict["layers"]["layer_1"]["n_neurons"]
    total_size += n_in * n_out + n_out
    
    # First hidden to second hidden: 32 * 16 + 16
    n_in = minimal_nn_dict["layers"]["layer_1"]["n_neurons"]
    n_out = minimal_nn_dict["layers"]["layer_2"]["n_neurons"]
    total_size += n_in * n_out + n_out
    
    # Second hidden to output: 16 * 100 + 100
    n_in = minimal_nn_dict["layers"]["layer_2"]["n_neurons"]
    n_out = minimal_nn_dict["n_output_features"]
    total_size += n_in * n_out + n_out
    
    # Return random weights with fixed seed for reproducibility
    np.random.seed(42)
    weights = np.random.randn(total_size) * 0.1
    return weights


@pytest.fixture
def sample_normalization_params():
    """Sample input and output normalization parameters."""
    # Input normalization for 6 cosmological parameters
    in_minmax = np.array([
        [2.5, 3.5],      # ln10As
        [0.88, 1.05],    # ns
        [40.0, 100.0],   # H0
        [0.019, 0.025],  # ωb
        [0.08, 0.20],    # ωc
        [0.02, 0.12]     # τ
    ])
    
    # Output normalization for 100 Cl values
    out_minmax = np.column_stack([
        np.zeros(100),
        np.ones(100) * 1000.0
    ])
    
    return in_minmax, out_minmax


@pytest.fixture
def sample_cosmological_params():
    """Standard test cosmological parameters within training bounds."""
    return np.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])


@pytest.fixture
def postprocessing_module_content():
    """Content for a test postprocessing module."""
    return """
import jax.numpy as jnp

def postprocessing(input_params, output):
    # Simple test postprocessing: scale by first parameter
    return output * jnp.exp(input_params[0] - 3.0)
"""


@pytest.fixture
def mock_emulator_directory(tmp_path, minimal_nn_dict, sample_weights, 
                           sample_normalization_params, postprocessing_module_content):
    """Create a temporary emulator directory with all required files."""
    emulator_dir = tmp_path / "test_emulator"
    emulator_dir.mkdir()
    
    # Write nn_setup.json
    with open(emulator_dir / "nn_setup.json", "w") as f:
        json.dump(minimal_nn_dict, f)
    
    # Write weights.npy
    np.save(emulator_dir / "weights.npy", sample_weights)
    
    # Write normalization parameters
    in_minmax, out_minmax = sample_normalization_params
    np.save(emulator_dir / "inminmax.npy", in_minmax)
    np.save(emulator_dir / "outminmax.npy", out_minmax)
    
    # Write postprocessing.py
    with open(emulator_dir / "postprocessing.py", "w") as f:
        f.write(postprocessing_module_content)
    
    # Write l.npy (multipole values)
    l_values = np.arange(2, 102)  # l from 2 to 101 for 100 outputs
    np.save(emulator_dir / "l.npy", l_values)
    
    return emulator_dir


@pytest.fixture
def incomplete_emulator_directory(tmp_path, minimal_nn_dict):
    """Create an emulator directory missing some required files."""
    emulator_dir = tmp_path / "incomplete_emulator"
    emulator_dir.mkdir()
    
    # Only write nn_setup.json
    with open(emulator_dir / "nn_setup.json", "w") as f:
        json.dump(minimal_nn_dict, f)
    
    return emulator_dir


@pytest.fixture
def corrupted_emulator_directory(tmp_path, sample_normalization_params):
    """Create an emulator directory with corrupted files."""
    emulator_dir = tmp_path / "corrupted_emulator"
    emulator_dir.mkdir()
    
    # Write invalid JSON
    with open(emulator_dir / "nn_setup.json", "w") as f:
        f.write("{ invalid json }")
    
    # Write corrupted numpy file
    with open(emulator_dir / "weights.npy", "wb") as f:
        f.write(b"corrupted data")
    
    # Add normalization files so we get to JSON parsing error
    in_minmax, out_minmax = sample_normalization_params
    np.save(emulator_dir / "inminmax.npy", in_minmax)
    np.save(emulator_dir / "outminmax.npy", out_minmax)
    
    # Add postprocessing file
    with open(emulator_dir / "postprocessing.py", "w") as f:
        f.write("def postprocessing(input, output): return output")
    
    return emulator_dir


@pytest.fixture
def invalid_postprocessing_directory(tmp_path, minimal_nn_dict, sample_weights,
                                    sample_normalization_params):
    """Create an emulator directory with invalid postprocessing."""
    emulator_dir = tmp_path / "invalid_postprocessing"
    emulator_dir.mkdir()
    
    # Write valid files
    with open(emulator_dir / "nn_setup.json", "w") as f:
        json.dump(minimal_nn_dict, f)
    
    np.save(emulator_dir / "weights.npy", sample_weights)
    
    in_minmax, out_minmax = sample_normalization_params
    np.save(emulator_dir / "inminmax.npy", in_minmax)
    np.save(emulator_dir / "outminmax.npy", out_minmax)
    
    # Write postprocessing with syntax error
    with open(emulator_dir / "postprocessing.py", "w") as f:
        f.write("def postprocessing(input, output):\n    return output * ")
    
    return emulator_dir


@pytest.fixture
def batch_cosmological_params(sample_cosmological_params):
    """Batch of cosmological parameters for testing batch processing."""
    # Create 10 slightly varied parameter sets
    batch_size = 10
    params = np.tile(sample_cosmological_params, (batch_size, 1))
    
    # Add small variations
    np.random.seed(42)
    variations = np.random.randn(batch_size, 6) * 0.01
    params += variations
    
    return params


@pytest.fixture
def edge_case_params():
    """Parameters at the edge of training bounds."""
    return {
        "lower_bound": np.array([2.5, 0.88, 40.0, 0.019, 0.08, 0.02]),
        "upper_bound": np.array([3.5, 1.05, 100.0, 0.025, 0.20, 0.12]),
        "out_of_bounds_low": np.array([2.0, 0.80, 30.0, 0.015, 0.05, 0.01]),
        "out_of_bounds_high": np.array([4.0, 1.10, 110.0, 0.030, 0.25, 0.15])
    }


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in all tests."""
    np.random.seed(42)
    # Note: JAX random seeds are handled separately in individual tests
    yield
    # Reset after test if needed