"""
Test inference capabilities of jaxcapse.

Focus on forward pass, batch processing, and output validation.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxcapse import jaxcapse
from tests.fixtures import *

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestGetCl:
    """Test single input inference with get_Cl method."""
    
    def test_get_cl_single_input(self, mock_emulator_directory, sample_cosmological_params):
        """Test single parameter set inference."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Run inference
        cl_values = mlp.get_Cl(sample_cosmological_params)
        
        # Check output exists
        assert cl_values is not None
        assert len(cl_values) > 0
    
    def test_get_cl_output_shape(self, mock_emulator_directory, sample_cosmological_params):
        """Verify output dimensions match expected."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        cl_values = mlp.get_Cl(sample_cosmological_params)
        
        # Should return 100 Cl values (as configured in minimal_nn_dict)
        assert cl_values.shape == (100,)
    
    def test_get_cl_deterministic(self, mock_emulator_directory, sample_cosmological_params):
        """Ensure same input gives same output."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Run twice with same input
        cl_values1 = mlp.get_Cl(sample_cosmological_params)
        cl_values2 = mlp.get_Cl(sample_cosmological_params)
        
        # Should be identical
        assert jnp.allclose(cl_values1, cl_values2, rtol=1e-10)
    
    def test_get_cl_jit_compilation(self, mock_emulator_directory, sample_cosmological_params):
        """Verify JIT compilation works."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Convert to JAX array to ensure JIT compilation
        params_jax = jnp.array(sample_cosmological_params)
        
        # First call (includes compilation)
        cl_values1 = mlp.get_Cl(params_jax)
        
        # Second call (uses compiled version)
        cl_values2 = mlp.get_Cl(params_jax)
        
        # Results should be identical
        assert jnp.allclose(cl_values1, cl_values2)
    
    def test_get_cl_finite_outputs(self, mock_emulator_directory, sample_cosmological_params):
        """Check no NaN/Inf in outputs."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        cl_values = mlp.get_Cl(sample_cosmological_params)
        
        # All values should be finite
        assert jnp.all(jnp.isfinite(cl_values))
        
        # No NaN values
        assert not jnp.any(jnp.isnan(cl_values))
        
        # No Inf values
        assert not jnp.any(jnp.isinf(cl_values))


class TestBatchProcessing:
    """Test batch inference capabilities."""
    
    def test_get_cl_batch_shape(self, mock_emulator_directory, batch_cosmological_params):
        """Verify batch output dimensions."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Process batch
        cl_batch = mlp.get_Cl_batch(batch_cosmological_params)
        
        # Check shape: (batch_size, n_cls)
        expected_shape = (batch_cosmological_params.shape[0], 100)
        assert cl_batch.shape == expected_shape
    
    def test_get_cl_batch_consistency(self, mock_emulator_directory, batch_cosmological_params):
        """Batch results match individual calls."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Process as batch
        cl_batch = mlp.get_Cl_batch(batch_cosmological_params)
        
        # Process individually
        cl_individual = []
        for params in batch_cosmological_params:
            cl_individual.append(mlp.get_Cl(params))
        cl_individual = jnp.array(cl_individual)
        
        # Should be identical
        assert jnp.allclose(cl_batch, cl_individual, rtol=1e-10)
    
    def test_get_cl_batch_empty(self, mock_emulator_directory):
        """Handle empty batch gracefully."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Empty batch
        empty_batch = np.array([]).reshape(0, 6)
        
        cl_batch = mlp.get_Cl_batch(empty_batch)
        
        # Should return empty array with correct shape
        assert cl_batch.shape == (0, 100)
    
    def test_get_cl_batch_single(self, mock_emulator_directory, sample_cosmological_params):
        """Single-item batch works correctly."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Single item batch
        single_batch = sample_cosmological_params.reshape(1, -1)
        
        cl_batch = mlp.get_Cl_batch(single_batch)
        
        # Check shape
        assert cl_batch.shape == (1, 100)
        
        # Should match single call
        cl_single = mlp.get_Cl(sample_cosmological_params)
        assert jnp.allclose(cl_batch[0], cl_single, rtol=1e-10)
    
    def test_get_cl_batch_large(self, mock_emulator_directory, sample_cosmological_params):
        """Test with large batches for memory efficiency."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Create large batch
        large_batch_size = 1000
        large_batch = np.tile(sample_cosmological_params, (large_batch_size, 1))
        
        # Add small random variations
        np.random.seed(42)
        large_batch += np.random.randn(*large_batch.shape) * 0.001
        
        # Process large batch
        cl_batch = mlp.get_Cl_batch(large_batch)
        
        # Check shape and finiteness
        assert cl_batch.shape == (large_batch_size, 100)
        assert jnp.all(jnp.isfinite(cl_batch))


class TestInputVariations:
    """Test inference with various input types and values."""
    
    def test_numpy_array_input(self, mock_emulator_directory, sample_cosmological_params):
        """Test with numpy array input."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Ensure input is numpy array
        params_np = np.array(sample_cosmological_params)
        cl_values = mlp.get_Cl(params_np)
        
        assert cl_values is not None
        assert cl_values.shape == (100,)
    
    def test_jax_array_input(self, mock_emulator_directory, sample_cosmological_params):
        """Test with JAX array input."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Convert to JAX array
        params_jax = jnp.array(sample_cosmological_params)
        cl_values = mlp.get_Cl(params_jax)
        
        assert cl_values is not None
        assert cl_values.shape == (100,)
    
    def test_list_input_not_supported(self, mock_emulator_directory, sample_cosmological_params):
        """Test that Python list input is not supported (users must use JAX arrays)."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Convert to list
        params_list = sample_cosmological_params.tolist()
        
        # Should raise an error since lists are not supported
        with pytest.raises(AttributeError):
            mlp.get_Cl(params_list)
    
    def test_edge_case_lower_bound(self, mock_emulator_directory, edge_case_params):
        """Test with parameters at lower training bound."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        cl_values = mlp.get_Cl(edge_case_params["lower_bound"])
        
        assert jnp.all(jnp.isfinite(cl_values))
        assert cl_values.shape == (100,)
    
    def test_edge_case_upper_bound(self, mock_emulator_directory, edge_case_params):
        """Test with parameters at upper training bound."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        cl_values = mlp.get_Cl(edge_case_params["upper_bound"])
        
        assert jnp.all(jnp.isfinite(cl_values))
        assert cl_values.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])