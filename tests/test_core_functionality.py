"""
Test core functionality of jaxcapse MLP emulator.

Focus on software functionality rather than cosmological accuracy.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxcapse import jaxcapse
from tests.fixtures import *

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)


class TestMLPInitialization:
    """Test MLP class initialization."""
    
    def test_mlp_init_with_valid_components(self, mock_emulator_directory):
        """Verify MLP initializes with all required components."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Check that MLP instance is created
        assert isinstance(mlp, jaxcapse.MLP)
        
        # Check required attributes exist
        assert mlp.emulator is not None
        assert mlp.in_MinMax is not None
        assert mlp.out_MinMax is not None
        assert mlp.postprocessing is not None
        assert mlp.emulator_description is not None
    
    def test_mlp_attributes_are_jax_arrays(self, mock_emulator_directory):
        """Ensure normalization parameters are JAX arrays."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Check that MinMax arrays are JAX arrays
        assert isinstance(mlp.in_MinMax, jnp.ndarray)
        assert isinstance(mlp.out_MinMax, jnp.ndarray)
    
    def test_no_backward_compatibility_attributes(self, mock_emulator_directory):
        """Verify backward compatibility attributes are removed."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # These attributes should not exist anymore
        assert not hasattr(mlp, 'NN_params')
        assert not hasattr(mlp, 'features')
        assert not hasattr(mlp, 'activations')
        assert not hasattr(mlp, 'apply')


class TestNormalization:
    """Test normalization is handled internally through jaxace."""
    
    def test_normalization_arrays_exist(self, mock_emulator_directory):
        """Verify normalization arrays are stored as JAX arrays."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Check that MinMax arrays exist and are JAX arrays
        assert mlp.in_MinMax is not None
        assert mlp.out_MinMax is not None
        assert isinstance(mlp.in_MinMax, jnp.ndarray)
        assert isinstance(mlp.out_MinMax, jnp.ndarray)
    
    def test_normalization_through_jaxace(self, mock_emulator_directory):
        """Test that normalization works through jaxace functions."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Test that we can use jaxace functions directly with the stored arrays
        from jaxace import maximin, inv_maximin
        
        # Test input normalization
        test_input = jnp.array([3.0, 0.95, 70.0, 0.022, 0.15, 0.08])
        normalized = maximin(test_input, mlp.in_MinMax)
        assert normalized.shape == test_input.shape
        
        # Test output normalization round trip
        test_output = np.random.uniform(0, 500, 100)
        norm_output = maximin(test_output, mlp.out_MinMax)
        recovered_output = inv_maximin(norm_output, mlp.out_MinMax)
        assert np.allclose(recovered_output, test_output, rtol=1e-10)
    
    def test_normalization_bounds(self, mock_emulator_directory, edge_case_params):
        """Ensure normalized values are in expected range using jaxace."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        from jaxace import maximin
        
        # Test lower bound
        lower_norm = maximin(edge_case_params["lower_bound"], mlp.in_MinMax)
        assert np.allclose(lower_norm, np.zeros(6), atol=1e-10)
        
        # Test upper bound
        upper_norm = maximin(edge_case_params["upper_bound"], mlp.in_MinMax)
        assert np.allclose(upper_norm, np.ones(6), atol=1e-10)


class TestEmulatorLoading:
    """Test emulator loading functionality."""
    
    def test_load_emulator_valid_path(self, mock_emulator_directory):
        """Load emulator from valid directory."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        assert isinstance(mlp, jaxcapse.MLP)
        
        # Check emulator description loaded correctly
        assert mlp.emulator_description["author"] == "Test Suite"
        assert "p1, p2, p3, p4, p5, p6" in mlp.emulator_description["parameters"]
    
    def test_load_emulator_missing_weights(self, incomplete_emulator_directory):
        """Handle missing weights file gracefully."""
        with pytest.raises(FileNotFoundError):
            jaxcapse.load_emulator(str(incomplete_emulator_directory))
    
    def test_load_emulator_corrupted_json(self, corrupted_emulator_directory):
        """Handle corrupted JSON file."""
        with pytest.raises(json.JSONDecodeError):
            jaxcapse.load_emulator(str(corrupted_emulator_directory))
    
    def test_load_preprocessing_function(self, mock_emulator_directory):
        """Verify postprocessing function loads correctly."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Check postprocessing is callable
        assert callable(mlp.postprocessing)
        
        # Test that it works with dummy data
        input_params = jnp.array([3.0, 0.96, 67, 0.022, 0.12, 0.055])
        output = jnp.ones(100)
        
        result = mlp.postprocessing(input_params, output)
        assert result.shape == output.shape
        assert jnp.all(jnp.isfinite(result))
    
    def test_folder_path_normalization(self, mock_emulator_directory):
        """Test path handling with/without trailing slash."""
        # Test without trailing slash
        mlp1 = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        # Test with trailing slash
        mlp2 = jaxcapse.load_emulator(str(mock_emulator_directory) + "/")
        
        # Both should work
        assert isinstance(mlp1, jaxcapse.MLP)
        assert isinstance(mlp2, jaxcapse.MLP)


class TestEmulatorDescription:
    """Test emulator description and metadata."""
    
    def test_emulator_description_fields(self, mock_emulator_directory):
        """Check all expected description fields are present."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        desc = mlp.emulator_description
        assert "author" in desc
        assert "author_email" in desc
        assert "parameters" in desc
        assert "miscellanea" in desc
    
    def test_emulator_description_content(self, mock_emulator_directory):
        """Verify description content matches configuration."""
        mlp = jaxcapse.load_emulator(str(mock_emulator_directory))
        
        desc = mlp.emulator_description
        assert desc["author"] == "Test Suite"
        assert desc["author_email"] == "test@example.com"
        assert desc["parameters"] == "p1, p2, p3, p4, p5, p6"
        assert desc["miscellanea"] == "Test emulator for unit testing"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])