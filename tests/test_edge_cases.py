"""
Test edge cases and error handling for jaxcapse.

Focus on boundary conditions and error scenarios.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxcapse import jaxcapse
from tests.fixtures import *

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_input_wrong_dimension(self, mock_emulator_directory):
        """Test with wrong number of parameters."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Too few parameters
        wrong_params = jnp.array([3.1, 0.96, 67.0])  # Only 3 instead of 6
        with pytest.raises(Exception):  # Will raise shape mismatch in neural network
            mlp.get_Cl(wrong_params)
        
        # Too many parameters
        wrong_params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055, 0.1])  # 7 instead of 6
        with pytest.raises(Exception):  # Will raise shape mismatch in neural network
            mlp.get_Cl(wrong_params)
    
    def test_input_nan_values(self, mock_emulator_directory, sample_cosmological_params):
        """Test handling of NaN in inputs."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Create input with NaN
        params_with_nan = sample_cosmological_params.copy()
        params_with_nan[2] = np.nan
        params_jax = jnp.array(params_with_nan)
        
        # Run inference - JAX will propagate NaN
        cl_values = mlp.get_Cl(params_jax)
        
        # Output should contain NaN due to NaN propagation
        assert jnp.any(jnp.isnan(cl_values))
    
    def test_input_inf_values(self, mock_emulator_directory, sample_cosmological_params):
        """Test handling of Inf in inputs."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Create input with Inf
        params_with_inf = sample_cosmological_params.copy()
        params_with_inf[1] = np.inf
        params_jax = jnp.array(params_with_inf)
        
        # Run inference - JAX handles Inf gracefully
        cl_values = mlp.get_Cl(params_jax)
        
        # Note: tanh activation saturates inf to 1, so output may still be finite
        # Just verify the computation completes without error
        assert cl_values is not None
        assert cl_values.shape == (100,)


class TestFileHandling:
    """Test file loading error scenarios."""
    
    def test_nonexistent_directory(self):
        """Test loading from non-existent directory."""
        with pytest.raises(FileNotFoundError):
            jaxcapse.load_emulator("/nonexistent/path/to/emulator")
    
    def test_empty_directory(self, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty_emulator"
        empty_dir.mkdir()
        
        with pytest.raises(FileNotFoundError):
            jaxcapse.load_emulator(str(empty_dir))
    
    def test_missing_each_required_file(self, tmp_path, minimal_nn_dict, 
                                       sample_weights, sample_normalization_params):
        """Test that each required file is actually required."""
        # Test missing nn_setup.json
        dir1 = tmp_path / "missing_nn_setup"
        dir1.mkdir()
        in_minmax, out_minmax = sample_normalization_params
        np.save(dir1 / "weights.npy", sample_weights)
        np.save(dir1 / "inminmax.npy", in_minmax)
        np.save(dir1 / "outminmax.npy", out_minmax)
        with open(dir1 / "postprocessing.py", "w") as f:
            f.write("def postprocessing(input, output): return output")
        
        with pytest.raises(FileNotFoundError):
            jaxcapse.load_emulator(str(dir1))
        
        # Test missing weights.npy
        dir2 = tmp_path / "missing_weights"
        dir2.mkdir()
        with open(dir2 / "nn_setup.json", "w") as f:
            json.dump(minimal_nn_dict, f)
        np.save(dir2 / "inminmax.npy", in_minmax)
        np.save(dir2 / "outminmax.npy", out_minmax)
        with open(dir2 / "postprocessing.py", "w") as f:
            f.write("def postprocessing(input, output): return output")
        
        with pytest.raises(FileNotFoundError):
            jaxcapse.load_emulator(str(dir2))
        
        # Test missing inminmax.npy
        dir3 = tmp_path / "missing_inminmax"
        dir3.mkdir()
        with open(dir3 / "nn_setup.json", "w") as f:
            json.dump(minimal_nn_dict, f)
        np.save(dir3 / "weights.npy", sample_weights)
        np.save(dir3 / "outminmax.npy", out_minmax)
        with open(dir3 / "postprocessing.py", "w") as f:
            f.write("def postprocessing(input, output): return output")
        
        with pytest.raises(FileNotFoundError):
            jaxcapse.load_emulator(str(dir3))
    
    def test_incompatible_weight_dimensions(self, tmp_path, minimal_nn_dict, 
                                           sample_normalization_params):
        """Test weights that don't match network architecture."""
        emulator_dir = tmp_path / "incompatible_weights"
        emulator_dir.mkdir()
        
        # Write config
        with open(emulator_dir / "nn_setup.json", "w") as f:
            json.dump(minimal_nn_dict, f)
        
        # Write wrong size weights
        wrong_weights = np.random.randn(100)  # Wrong size
        np.save(emulator_dir / "weights.npy", wrong_weights)
        
        # Write normalization
        in_minmax, out_minmax = sample_normalization_params
        np.save(emulator_dir / "inminmax.npy", in_minmax)
        np.save(emulator_dir / "outminmax.npy", out_minmax)
        
        # Write postprocessing
        with open(emulator_dir / "postprocessing.py", "w") as f:
            f.write("def postprocessing(input, output): return output")
        
        # Should fail during emulator initialization
        with pytest.raises(ValueError):
            jaxcapse.load_emulator(str(emulator_dir))


class TestPostprocessingErrors:
    """Test postprocessing error scenarios."""
    
    def test_postprocessing_missing_function(self, tmp_path, minimal_nn_dict,
                                            sample_weights, sample_normalization_params):
        """Test postprocessing module without required function."""
        emulator_dir = tmp_path / "bad_postprocessing"
        emulator_dir.mkdir()
        
        # Write all files
        with open(emulator_dir / "nn_setup.json", "w") as f:
            json.dump(minimal_nn_dict, f)
        np.save(emulator_dir / "weights.npy", sample_weights)
        
        in_minmax, out_minmax = sample_normalization_params
        np.save(emulator_dir / "inminmax.npy", in_minmax)
        np.save(emulator_dir / "outminmax.npy", out_minmax)
        
        # Write postprocessing without the function
        with open(emulator_dir / "postprocessing.py", "w") as f:
            f.write("def other_function(x): return x")
        
        # Should fail when trying to access postprocessing function
        with pytest.raises(AttributeError):
            jaxcapse.load_emulator(str(emulator_dir))
    
    def test_postprocessing_syntax_error(self, invalid_postprocessing_directory):
        """Test postprocessing with Python syntax error."""
        with pytest.raises(SyntaxError):
            jaxcapse.load_emulator(str(invalid_postprocessing_directory))


class TestBatchEdgeCases:
    """Test edge cases specific to batch processing."""
    
    def test_very_large_batch_memory(self, mock_emulator_directory):
        """Test memory handling with very large batch."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Create a very large batch (but not so large to OOM)
        large_batch_size = 10000
        base_params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])
        large_batch = jnp.tile(base_params, (large_batch_size, 1))
        
        # Add small variations to make each unique
        key = jax.random.PRNGKey(42)
        noise = jax.random.normal(key, shape=large_batch.shape) * 0.001
        large_batch = large_batch + noise
        
        # Should handle large batch efficiently
        cl_batch = mlp.get_Cl_batch(large_batch)
        
        assert cl_batch.shape == (large_batch_size, 100)
        assert jnp.all(jnp.isfinite(cl_batch))
    
    def test_batch_with_identical_inputs(self, mock_emulator_directory, sample_cosmological_params):
        """Test batch where all inputs are identical."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Create batch with identical inputs
        batch_size = 5
        identical_batch = jnp.tile(sample_cosmological_params, (batch_size, 1))
        
        cl_batch = mlp.get_Cl_batch(identical_batch)
        
        # All outputs should be identical
        for i in range(1, batch_size):
            assert jnp.allclose(cl_batch[0], cl_batch[i], rtol=1e-10)


class TestDTypeHandling:
    """Test handling of different data types."""
    
    def test_float32_input(self, mock_emulator_directory, sample_cosmological_params):
        """Test with float32 input (should work with JAX)."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Convert to float32
        params_f32 = jnp.array(sample_cosmological_params, dtype=jnp.float32)
        
        # Should work (JAX handles dtype conversion)
        cl_values = mlp.get_Cl(params_f32)
        assert cl_values is not None
        assert jnp.all(jnp.isfinite(cl_values))
    
    def test_int_input(self, mock_emulator_directory):
        """Test with integer input (should be converted by JAX)."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Use integers (will be converted to float by JAX)
        params_int = jnp.array([3, 1, 67, 0, 0, 0])
        
        # Should work with automatic conversion
        cl_values = mlp.get_Cl(params_int)
        assert cl_values is not None
        assert jnp.all(jnp.isfinite(cl_values))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])