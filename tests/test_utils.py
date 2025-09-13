"""
Test utility functions in jaxcapse.

Focus on load_preprocessing and other utility functions.
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaxcapse.jaxcapse import load_preprocessing


class TestLoadPreprocessing:
    """Test the load_preprocessing function."""
    
    def test_load_valid_preprocessing(self, tmp_path):
        """Test loading a valid preprocessing module."""
        # Create a valid postprocessing module
        module_content = """
import jax.numpy as jnp

def postprocessing(input_params, output):
    return output * 2.0
"""
        # Write to file
        module_path = tmp_path / "postprocessing.py"
        with open(module_path, "w") as f:
            f.write(module_content)
        
        # Load the module
        postproc_func = load_preprocessing(str(tmp_path), "postprocessing")
        
        # Verify it's callable
        assert callable(postproc_func)
        
        # Test it works
        import jax.numpy as jnp
        test_input = jnp.array([1, 2, 3])
        test_output = jnp.array([4, 5, 6])
        result = postproc_func(test_input, test_output)
        assert jnp.allclose(result, test_output * 2.0)
    
    def test_load_preprocessing_missing_file(self, tmp_path):
        """Test loading from non-existent file."""
        # Try to load non-existent module
        with pytest.raises(FileNotFoundError):
            load_preprocessing(str(tmp_path), "nonexistent")
    
    def test_load_preprocessing_syntax_error(self, tmp_path):
        """Test loading module with syntax error."""
        # Create module with syntax error
        module_content = "def postprocessing(input, output):\n    return output *"
        
        module_path = tmp_path / "bad_syntax.py"
        with open(module_path, "w") as f:
            f.write(module_content)
        
        # Should raise SyntaxError
        with pytest.raises(SyntaxError):
            load_preprocessing(str(tmp_path), "bad_syntax")
    
    def test_load_preprocessing_missing_function(self, tmp_path):
        """Test loading module without postprocessing function."""
        # Create module without postprocessing function
        module_content = """
def other_function(x):
    return x * 2
"""
        module_path = tmp_path / "no_postproc.py"
        with open(module_path, "w") as f:
            f.write(module_content)
        
        # Should raise AttributeError when trying to access postprocessing
        with pytest.raises(AttributeError):
            load_preprocessing(str(tmp_path), "no_postproc")
    
    def test_load_preprocessing_with_imports(self, tmp_path):
        """Test loading module with external imports."""
        # Create module with imports
        module_content = """
import numpy as np
import jax.numpy as jnp

def postprocessing(input_params, output):
    # Use both numpy and jax
    factor = np.array(2.0)
    return output * jnp.array(factor)
"""
        module_path = tmp_path / "with_imports.py"
        with open(module_path, "w") as f:
            f.write(module_content)
        
        # Should load successfully
        postproc_func = load_preprocessing(str(tmp_path), "with_imports")
        assert callable(postproc_func)
        
        # Test it works
        import jax.numpy as jnp
        test_input = jnp.array([1, 2, 3])
        test_output = jnp.array([4, 5, 6])
        result = postproc_func(test_input, test_output)
        assert jnp.allclose(result, test_output * 2.0)
    
    def test_load_preprocessing_callable(self, tmp_path):
        """Test that loaded preprocessing is callable."""
        # Create a valid module
        module_content = """
import jax.numpy as jnp

def postprocessing(input_params, output):
    return output * jnp.exp(input_params[0])
"""
        module_path = tmp_path / "valid_proc.py"
        with open(module_path, "w") as f:
            f.write(module_content)
        
        # Loading succeeds
        postproc_func = load_preprocessing(str(tmp_path), "valid_proc")
        assert callable(postproc_func)
        
        # Test it can be called successfully
        import jax.numpy as jnp
        test_input = jnp.array([1.0, 2.0, 3.0])
        test_output = jnp.array([4.0, 5.0, 6.0])
        
        result = postproc_func(test_input, test_output)
        expected = test_output * jnp.exp(test_input[0])
        assert jnp.allclose(result, expected)


class TestIntegrationWithRealEmulator:
    """Test that the actual trained emulators can be loaded."""
    
    def test_load_real_emulator_if_exists(self):
        """Test loading actual trained emulator if available."""
        # Check if trained_emu directory exists
        trained_path = Path(__file__).parent.parent / "trained_emu" / "TT"
        
        if trained_path.exists():
            from jaxcapse import jaxcapse
            # Try to load the real emulator
            mlp = jaxcapse.load_emulator(str(trained_path))
            
            # Basic checks
            assert mlp is not None
            assert hasattr(mlp, 'get_Cl')
            assert hasattr(mlp, 'emulator_description')
            
            # Test inference with realistic parameters
            import jax.numpy as jnp
            params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])
            cl = mlp.get_Cl(params)
            
            # Check output is reasonable
            assert cl.shape[0] > 0  # Has some Cl values
            assert jnp.all(jnp.isfinite(cl))  # All finite
        else:
            pytest.skip("trained_emu directory not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])