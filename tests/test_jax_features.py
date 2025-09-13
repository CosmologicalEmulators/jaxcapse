"""
Test JAX-specific features of jaxcapse.

Focus on JIT compilation, gradients, and vectorization.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxcapse import jaxcapse
from tests.fixtures import *

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestJITCompilation:
    """Test JIT compilation features."""
    
    def test_jit_compilation_speedup(self, mock_emulator_directory, sample_cosmological_params):
        """Verify JIT compilation provides speedup after first call."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        # First call (includes compilation)
        _ = mlp.get_Cl(params_jax)
        
        # Measure time for subsequent calls
        start = time.perf_counter()
        for _ in range(10):
            _ = mlp.get_Cl(params_jax)
        jit_time = time.perf_counter() - start
        
        # JIT should be fast (this is a simple test, not a rigorous benchmark)
        assert jit_time < 1.0  # Should complete 10 calls in under 1 second
    
    def test_jit_with_different_values_same_shape(self, mock_emulator_directory):
        """Test JIT works with different values but same shape."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Different values, same shape
        params1 = jnp.array([3.0, 0.95, 65.0, 0.021, 0.11, 0.05])
        params2 = jnp.array([3.2, 0.97, 70.0, 0.023, 0.13, 0.06])
        
        cl1 = mlp.get_Cl(params1)
        cl2 = mlp.get_Cl(params2)
        
        # Should get different results
        assert not jnp.allclose(cl1, cl2)
        # But same shape
        assert cl1.shape == cl2.shape


class TestGradients:
    """Test gradient computation through the emulator."""
    
    def test_gradient_basic(self, mock_emulator_directory, sample_cosmological_params):
        """Test basic gradient computation."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        # Define a scalar loss function
        def loss_fn(params):
            cl = mlp.get_Cl(params)
            return jnp.sum(cl ** 2)
        
        # Compute gradient
        grad = jax.grad(loss_fn)(params_jax)
        
        # Check gradient shape and finiteness
        assert grad.shape == params_jax.shape
        assert jnp.all(jnp.isfinite(grad))
    
    def test_jacobian_forward(self, mock_emulator_directory, sample_cosmological_params):
        """Test Jacobian computation using forward mode."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        # Compute Jacobian (forward mode is efficient when n_outputs > n_inputs)
        jacobian = jax.jacfwd(mlp.get_Cl)(params_jax)
        
        # Check shape: (n_outputs, n_inputs) = (100, 6)
        assert jacobian.shape == (100, 6)
        assert jnp.all(jnp.isfinite(jacobian))
    
    def test_jacobian_reverse(self, mock_emulator_directory, sample_cosmological_params):
        """Test Jacobian computation using reverse mode."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        # Compute Jacobian (reverse mode, less efficient for this case but should work)
        jacobian = jax.jacrev(mlp.get_Cl)(params_jax)
        
        # Check shape: (n_outputs, n_inputs) = (100, 6)
        assert jacobian.shape == (100, 6)
        assert jnp.all(jnp.isfinite(jacobian))
    
    def test_hessian(self, mock_emulator_directory, sample_cosmological_params):
        """Test Hessian (second derivative) computation."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        # Define scalar function for Hessian
        def scalar_fn(params):
            cl = mlp.get_Cl(params)
            return jnp.sum(cl)
        
        # Compute Hessian
        hessian = jax.hessian(scalar_fn)(params_jax)
        
        # Check shape: (n_params, n_params) = (6, 6)
        assert hessian.shape == (6, 6)
        assert jnp.all(jnp.isfinite(hessian))
        
        # Hessian should be symmetric (approximately)
        assert jnp.allclose(hessian, hessian.T, rtol=1e-10)


class TestVectorization:
    """Test vectorization with vmap."""
    
    def test_vmap_basic(self, mock_emulator_directory, batch_cosmological_params):
        """Test basic vmap functionality."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Vectorize get_Cl
        vmap_get_cl = jax.vmap(mlp.get_Cl)
        
        # Apply to batch
        cl_batch = vmap_get_cl(batch_cosmological_params)
        
        # Check shape
        assert cl_batch.shape == (batch_cosmological_params.shape[0], 100)
        assert jnp.all(jnp.isfinite(cl_batch))
    
    def test_vmap_vs_batch_method(self, mock_emulator_directory, batch_cosmological_params):
        """Compare vmap with built-in batch method."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Using vmap
        vmap_get_cl = jax.vmap(mlp.get_Cl)
        cl_vmap = vmap_get_cl(batch_cosmological_params)
        
        # Using built-in batch method
        cl_batch = mlp.get_Cl_batch(batch_cosmological_params)
        
        # Should be identical
        assert jnp.allclose(cl_vmap, cl_batch, rtol=1e-10)
    
    def test_vmap_with_grad(self, mock_emulator_directory, batch_cosmological_params):
        """Test combining vmap with gradient computation."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Define per-sample loss
        def loss_fn(params):
            cl = mlp.get_Cl(params)
            return jnp.sum(cl ** 2)
        
        # Vectorize gradient computation
        vmap_grad = jax.vmap(jax.grad(loss_fn))
        
        # Compute gradients for batch
        grads = vmap_grad(batch_cosmological_params)
        
        # Check shape: (batch_size, n_params)
        assert grads.shape == batch_cosmological_params.shape
        assert jnp.all(jnp.isfinite(grads))


class TestJAXTransformations:
    """Test various JAX transformations."""
    
    def test_value_and_grad(self, mock_emulator_directory, sample_cosmological_params):
        """Test value_and_grad for efficient computation."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        def loss_fn(params):
            cl = mlp.get_Cl(params)
            return jnp.mean(cl ** 2)
        
        # Get both value and gradient in one call
        value, grad = jax.value_and_grad(loss_fn)(params_jax)
        
        # Check results
        assert jnp.isfinite(value)
        assert grad.shape == params_jax.shape
        assert jnp.all(jnp.isfinite(grad))
    
    def test_jit_grad_composition(self, mock_emulator_directory, sample_cosmological_params):
        """Test composing JIT with grad."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        params_jax = jnp.array(sample_cosmological_params)
        
        def loss_fn(params):
            cl = mlp.get_Cl(params)
            return jnp.sum(cl)
        
        # JIT the gradient function
        grad_fn = jax.jit(jax.grad(loss_fn))
        
        # Compute gradient
        grad = grad_fn(params_jax)
        
        # Check result
        assert grad.shape == params_jax.shape
        assert jnp.all(jnp.isfinite(grad))
    
    def test_partial_application(self, mock_emulator_directory):
        """Test partial application of parameters."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Fix some parameters
        fixed_params = jnp.array([3.1, 0.96])
        
        def partial_fn(varying_params):
            full_params = jnp.concatenate([fixed_params, varying_params])
            return mlp.get_Cl(full_params)
        
        # Vary only last 4 parameters
        varying = jnp.array([67.0, 0.022, 0.12, 0.055])
        
        # Should work
        cl = partial_fn(varying)
        assert cl.shape == (100,)
        assert jnp.all(jnp.isfinite(cl))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])