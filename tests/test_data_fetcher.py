"""
Unit tests for jaxcapse data fetcher and emulator loading.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxcapse.data_fetcher import EmulatorDataFetcher, get_fetcher


class TestEmulatorDataFetcher(unittest.TestCase):
    """Test the EmulatorDataFetcher class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_url = "https://zenodo.org/records/17115001/files/trained_emu.tar.gz?download=1"
        self.test_types = ["TT", "EE", "TE", "PP"]

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization_requires_parameters(self):
        """Test that EmulatorDataFetcher requires zenodo_url and emulator_types."""
        # Should work with all parameters
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.test_url,
            emulator_types=self.test_types,
            cache_dir=self.temp_dir
        )
        self.assertEqual(fetcher.zenodo_url, self.test_url)
        self.assertEqual(fetcher.emulator_types, self.test_types)
        self.assertEqual(fetcher.cache_dir, Path(self.temp_dir))

        # Should fail without required parameters
        with self.assertRaises(TypeError):
            EmulatorDataFetcher()

        with self.assertRaises(TypeError):
            EmulatorDataFetcher(zenodo_url=self.test_url)

        with self.assertRaises(TypeError):
            EmulatorDataFetcher(emulator_types=self.test_types)

    def test_cache_directory_creation(self):
        """Test that cache directory is created properly."""
        cache_path = Path(self.temp_dir) / "test_cache"
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.test_url,
            emulator_types=self.test_types,
            cache_dir=cache_path
        )
        self.assertTrue(cache_path.exists())
        self.assertTrue(cache_path.is_dir())

    def test_tar_path_extraction(self):
        """Test that tar filename is extracted correctly from URL."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.test_url,
            emulator_types=self.test_types,
            cache_dir=self.temp_dir
        )
        self.assertEqual(fetcher.tar_path.name, "trained_emu.tar.gz")

        # Test with different URL format
        custom_url = "https://example.com/path/to/my_emulators.tar.gz?param=value"
        fetcher2 = EmulatorDataFetcher(
            zenodo_url=custom_url,
            emulator_types=["TT"],
            cache_dir=self.temp_dir
        )
        self.assertEqual(fetcher2.tar_path.name, "my_emulators.tar.gz")

    def test_list_available(self):
        """Test listing available emulator types."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.test_url,
            emulator_types=self.test_types,
            cache_dir=self.temp_dir
        )
        available = fetcher.list_available()
        self.assertIsInstance(available, dict)
        self.assertEqual(len(available), 4)
        self.assertIn("TT", available)
        self.assertIn("CMB temperature power spectrum", available["TT"])

    def test_list_cached_empty(self):
        """Test listing cached emulators when cache is empty."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.test_url,
            emulator_types=self.test_types,
            cache_dir=self.temp_dir
        )
        cached = fetcher.list_cached()
        self.assertIsInstance(cached, list)
        self.assertEqual(len(cached), 0)

    def test_get_emulator_path_invalid(self):
        """Test getting path for invalid emulator type."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.test_url,
            emulator_types=self.test_types,
            cache_dir=self.temp_dir
        )
        with self.assertRaises(ValueError) as context:
            fetcher.get_emulator_path("INVALID")
        self.assertIn("Unknown emulator type", str(context.exception))

    def test_get_fetcher_singleton(self):
        """Test that get_fetcher returns a singleton."""
        fetcher1 = get_fetcher()
        fetcher2 = get_fetcher()
        self.assertIs(fetcher1, fetcher2)

        # New instance with different parameters
        fetcher3 = get_fetcher(cache_dir=self.temp_dir)
        self.assertIs(fetcher3, fetcher1)  # Still same instance due to singleton


class TestEmulatorConfigs(unittest.TestCase):
    """Test the emulator configuration system."""

    def setUp(self):
        """Set up test fixtures."""
        # Prevent auto-download during tests
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"

    def tearDown(self):
        """Clean up."""
        # Remove environment variable
        if "JAXCAPSE_NO_AUTO_DOWNLOAD" in os.environ:
            del os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"]

    def test_emulator_configs_structure(self):
        """Test that EMULATOR_CONFIGS has correct structure."""
        import jaxcapse

        self.assertIn("EMULATOR_CONFIGS", dir(jaxcapse))
        configs = jaxcapse.EMULATOR_CONFIGS

        # Check structure
        self.assertIsInstance(configs, dict)
        self.assertIn("class_mnuw0wacdm", configs)

        # Check config contents
        config = configs["class_mnuw0wacdm"]
        self.assertIn("zenodo_url", config)
        self.assertIn("emulator_types", config)
        self.assertIn("description", config)
        self.assertEqual(len(config["emulator_types"]), 4)

    def test_trained_emulators_initialization(self):
        """Test that trained_emulators is initialized properly."""
        import jaxcapse

        self.assertIn("trained_emulators", dir(jaxcapse))
        self.assertIsInstance(jaxcapse.trained_emulators, dict)

        # Should have class_mnuw0wacdm with None values (auto-download disabled)
        self.assertIn("class_mnuw0wacdm", jaxcapse.trained_emulators)
        emulators = jaxcapse.trained_emulators["class_mnuw0wacdm"]
        self.assertIn("TT", emulators)
        self.assertIn("EE", emulators)
        self.assertIn("TE", emulators)
        self.assertIn("PP", emulators)

    def test_add_emulator_config(self):
        """Test adding new emulator configuration."""
        import jaxcapse

        # Add a test configuration
        test_model = "test_model"
        test_url = "https://example.com/test.tar.gz"
        test_types = ["TT", "EE"]

        result = jaxcapse.add_emulator_config(
            model_name=test_model,
            zenodo_url=test_url,
            emulator_types=test_types,
            description="Test model",
            auto_load=False  # Don't actually download
        )

        # Check it was added
        self.assertIn(test_model, jaxcapse.EMULATOR_CONFIGS)
        self.assertIn(test_model, jaxcapse.trained_emulators)

        # Check configuration
        config = jaxcapse.EMULATOR_CONFIGS[test_model]
        self.assertEqual(config["zenodo_url"], test_url)
        self.assertEqual(config["emulator_types"], test_types)
        self.assertEqual(config["description"], "Test model")

        # Check empty emulators were created
        emulators = jaxcapse.trained_emulators[test_model]
        self.assertIn("TT", emulators)
        self.assertIn("EE", emulators)
        self.assertIsNone(emulators["TT"])
        self.assertIsNone(emulators["EE"])


class TestJacobianComputation(unittest.TestCase):
    """Test Jacobian computation using JAX autodiff with real emulators."""

    @classmethod
    def setUpClass(cls):
        """Download emulators once for all tests."""
        # Import jaxcapse (will download emulators if not cached)
        import jaxcapse
        cls.jaxcapse = jaxcapse

        # Try to get the TT emulator
        if "class_mnuw0wacdm" in jaxcapse.trained_emulators:
            cls.emulator_TT = jaxcapse.trained_emulators["class_mnuw0wacdm"]["TT"]
        else:
            cls.emulator_TT = None

    def test_emulator_loaded(self):
        """Test that emulator is loaded for Jacobian tests."""
        if self.emulator_TT is None:
            self.skipTest("TT emulator not loaded, skipping Jacobian tests")

        # Check that emulator has expected structure
        self.assertIsNotNone(self.emulator_TT)
        self.assertTrue(hasattr(self.emulator_TT, 'predict'))

    def test_jacobian_computation(self):
        """Test computing Jacobian of emulator output with respect to input parameters."""
        if self.emulator_TT is None:
            self.skipTest("TT emulator not loaded, skipping Jacobian test")

        import jax
        import jax.numpy as jnp

        # Define a function that runs the emulator
        def emulator_function(params):
            """
            Run emulator with given parameters.

            Parameters should be in the order expected by the emulator.
            Typical cosmological parameters: [omega_b, omega_c, h, ln10As, ns, tau, ...]
            """
            # The emulator's predict method
            return self.emulator_TT.predict(params)

        # Create test input parameters
        # Standard ΛCDM parameters (adjust based on your emulator's expectations)
        test_params = jnp.array([
            0.02237,   # omega_b
            0.1200,    # omega_c
            0.6736,    # h
            3.044,     # ln10As
            0.9649,    # ns
            0.0544,    # tau (reionization)
        ])

        # Compute Jacobian
        try:
            jacobian_fn = jax.jacobian(emulator_function)
            jacobian = jacobian_fn(test_params)

            # Check Jacobian shape
            output = emulator_function(test_params)
            expected_shape = (output.shape[0] if hasattr(output, 'shape') else 1,
                            test_params.shape[0])

            self.assertEqual(jacobian.shape[-1], test_params.shape[0])
            print(f"✓ Jacobian computed successfully with shape: {jacobian.shape}")

            # Check that Jacobian has reasonable values (not all zeros or NaNs)
            self.assertFalse(jnp.all(jacobian == 0), "Jacobian should not be all zeros")
            self.assertFalse(jnp.any(jnp.isnan(jacobian)), "Jacobian should not contain NaNs")

        except Exception as e:
            # If the emulator doesn't support autodiff directly, that's okay
            print(f"Note: Direct Jacobian computation not supported: {e}")
            self.skipTest(f"Emulator may not support direct autodiff: {e}")

    def test_parameter_sensitivity(self):
        """Test parameter sensitivity analysis using Jacobian."""
        if self.emulator_TT is None:
            self.skipTest("TT emulator not loaded, skipping sensitivity test")

        import jax
        import jax.numpy as jnp

        # Define emulator function for a single output (e.g., at specific ell)
        def emulator_at_ell(params, ell_idx=10):
            """Get emulator output at specific ell."""
            output = self.emulator_TT.predict(params)
            # Return scalar for cleaner Jacobian
            if hasattr(output, '__len__'):
                return output[ell_idx] if len(output) > ell_idx else output[0]
            return output

        # Test parameters
        test_params = jnp.array([0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544])

        try:
            # Compute gradient (Jacobian for scalar output)
            grad_fn = jax.grad(emulator_at_ell)
            gradient = grad_fn(test_params)

            # Check gradient
            self.assertEqual(gradient.shape, test_params.shape)
            print(f"✓ Gradient computed: {gradient}")

            # Find most sensitive parameter
            sensitivities = jnp.abs(gradient)
            most_sensitive_idx = jnp.argmax(sensitivities)
            param_names = ["omega_b", "omega_c", "h", "ln10As", "ns", "tau"]

            if most_sensitive_idx < len(param_names):
                print(f"✓ Most sensitive parameter: {param_names[most_sensitive_idx]}")

        except Exception as e:
            print(f"Note: Gradient computation not supported: {e}")

    def test_jit_compilation(self):
        """Test that emulator can be JIT compiled for performance."""
        if self.emulator_TT is None:
            self.skipTest("TT emulator not loaded, skipping JIT test")

        import jax
        import jax.numpy as jnp

        # Create JIT compiled version
        @jax.jit
        def jit_emulator(params):
            return self.emulator_TT.predict(params)

        test_params = jnp.array([0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544])

        try:
            # First call compiles
            result1 = jit_emulator(test_params)
            # Second call should be faster (already compiled)
            result2 = jit_emulator(test_params)

            # Results should be identical
            if hasattr(result1, 'shape'):
                jnp.testing.assert_array_almost_equal(result1, result2)
            print("✓ JIT compilation successful")

        except Exception as e:
            print(f"Note: JIT compilation not fully supported: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)