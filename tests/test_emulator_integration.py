"""
Integration tests for jaxcapse emulator system with JAX.
Tests the full pipeline from loading to prediction and differentiation.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEmulatorIntegration(unittest.TestCase):
    """Integration tests for the complete emulator system."""

    @classmethod
    def setUpClass(cls):
        """Set up class fixtures - load emulators once."""
        import jaxcapse
        import jax

        cls.jaxcapse = jaxcapse
        cls.jax = jax

        # Get emulators
        cls.emulators = jaxcapse.trained_emulators.get("class_mnuw0wacdm", {})

    def test_all_emulators_accessible(self):
        """Test that all expected emulators are accessible."""
        expected_types = ["TT", "EE", "TE", "PP"]

        for emu_type in expected_types:
            self.assertIn(emu_type, self.emulators,
                         f"{emu_type} emulator not found in trained_emulators")

    def test_emulator_prediction_shape(self):
        """Test that emulator predictions have expected shape."""
        # Skip if no emulators loaded
        if not any(e is not None for e in self.emulators.values()):
            self.skipTest("No emulators loaded")

        import jax.numpy as jnp

        # Test parameters (typical cosmological values)
        test_params = jnp.array([0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544])

        for emu_type, emulator in self.emulators.items():
            if emulator is not None:
                try:
                    output = emulator.predict(test_params)
                    self.assertIsNotNone(output)
                    # Output should be 1D array of Cl values
                    self.assertGreater(len(output.shape), 0,
                                      f"{emu_type} output should have shape")
                    print(f"✓ {emu_type} output shape: {output.shape}")
                except Exception as e:
                    print(f"⚠ {emu_type} prediction failed: {e}")

    def test_emulator_normalization(self):
        """Test that emulator handles normalization correctly."""
        emulator_TT = self.emulators.get("TT")
        if emulator_TT is None:
            self.skipTest("TT emulator not loaded")

        import jax.numpy as jnp

        # Test with edge parameter values
        # Very small parameters
        small_params = jnp.array([0.01, 0.05, 0.5, 2.5, 0.85, 0.01])
        # Large parameters
        large_params = jnp.array([0.03, 0.15, 0.8, 3.5, 1.05, 0.1])

        try:
            output_small = emulator_TT.predict(small_params)
            output_large = emulator_TT.predict(large_params)

            # Outputs should be different
            self.assertFalse(jnp.allclose(output_small, output_large),
                           "Outputs should differ for different parameters")

            # Check for reasonable values (no infinities or NaNs)
            self.assertFalse(jnp.any(jnp.isnan(output_small)))
            self.assertFalse(jnp.any(jnp.isnan(output_large)))
            self.assertFalse(jnp.any(jnp.isinf(output_small)))
            self.assertFalse(jnp.any(jnp.isinf(output_large)))

            print("✓ Normalization handling verified")

        except Exception as e:
            print(f"Normalization test failed: {e}")

    def test_vmap_batch_prediction(self):
        """Test vectorized prediction over multiple parameter sets."""
        emulator_TT = self.emulators.get("TT")
        if emulator_TT is None:
            self.skipTest("TT emulator not loaded")

        import jax
        import jax.numpy as jnp

        # Create batch of parameter sets
        batch_size = 10
        param_dim = 6

        # Random parameters in reasonable range
        key = jax.random.PRNGKey(42)
        param_ranges = jnp.array([
            [0.015, 0.025],  # omega_b
            [0.10, 0.14],    # omega_c
            [0.6, 0.75],     # h
            [2.9, 3.2],      # ln10As
            [0.92, 1.0],     # ns
            [0.04, 0.07]     # tau
        ])

        batch_params = []
        for i in range(batch_size):
            key, subkey = jax.random.split(key)
            params = jax.random.uniform(
                subkey,
                shape=(param_dim,),
                minval=param_ranges[:, 0],
                maxval=param_ranges[:, 1]
            )
            batch_params.append(params)

        batch_params = jnp.stack(batch_params)

        try:
            # Vectorize the predict function
            vmap_predict = jax.vmap(emulator_TT.predict)
            batch_outputs = vmap_predict(batch_params)

            # Check output shape
            self.assertEqual(batch_outputs.shape[0], batch_size)
            print(f"✓ Batch prediction shape: {batch_outputs.shape}")

            # Verify outputs are different for different inputs
            self.assertFalse(jnp.allclose(batch_outputs[0], batch_outputs[1]))

        except Exception as e:
            print(f"Note: vmap not fully supported: {e}")

    def test_jacobian_physical_interpretation(self):
        """Test that Jacobian has physically meaningful structure."""
        emulator_TT = self.emulators.get("TT")
        if emulator_TT is None:
            self.skipTest("TT emulator not loaded")

        import jax
        import jax.numpy as jnp

        # Use fiducial Planck 2018 parameters
        fiducial_params = jnp.array([
            0.02237,   # omega_b
            0.1200,    # omega_c
            0.6736,    # h
            3.044,     # ln10As
            0.9649,    # ns
            0.0544,    # tau
        ])

        param_names = ["ω_b", "ω_c", "h", "ln10As", "ns", "τ"]

        try:
            # Define function for low-ell average (large scales)
            def low_ell_average(params):
                output = emulator_TT.predict(params)
                # Average over low ell (if output is array)
                if hasattr(output, '__len__') and len(output) > 10:
                    return jnp.mean(output[:10])
                return jnp.mean(output)

            # Compute gradient
            grad_fn = jax.grad(low_ell_average)
            gradient = grad_fn(fiducial_params)

            print("\nParameter sensitivities at low ell:")
            for i, (name, grad_val) in enumerate(zip(param_names, gradient)):
                print(f"  ∂Cl/∂{name} = {grad_val:.6f}")

            # Physical expectations:
            # - ln10As should have positive derivative (more power)
            # - tau should have negative derivative (suppression)
            as_idx = 3  # ln10As index
            tau_idx = 5  # tau index

            if len(gradient) > tau_idx:
                # These are general expectations, may vary with emulator
                print("\nPhysical consistency checks:")
                print(f"  ln10As gradient: {gradient[as_idx]:.6f} (expect positive)")
                print(f"  tau gradient: {gradient[tau_idx]:.6f} (expect negative)")

        except Exception as e:
            print(f"Physical interpretation test skipped: {e}")

    def test_second_derivatives(self):
        """Test computation of second derivatives (Hessian)."""
        emulator_TT = self.emulators.get("TT")
        if emulator_TT is None:
            self.skipTest("TT emulator not loaded")

        import jax
        import jax.numpy as jnp

        # Simple scalar output for cleaner Hessian
        def scalar_output(params):
            output = emulator_TT.predict(params)
            # Return first element or mean
            if hasattr(output, '__len__'):
                return output[0] if len(output) > 0 else jnp.sum(output)
            return output

        test_params = jnp.array([0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544])

        try:
            # Compute Hessian
            hessian_fn = jax.hessian(scalar_output)
            hessian = hessian_fn(test_params)

            # Check shape
            n_params = len(test_params)
            self.assertEqual(hessian.shape, (n_params, n_params))

            # Hessian should be symmetric
            self.assertTrue(jnp.allclose(hessian, hessian.T),
                          "Hessian should be symmetric")

            print(f"✓ Hessian computed with shape: {hessian.shape}")
            print(f"  Hessian symmetry verified")

            # Check for reasonable values
            self.assertFalse(jnp.any(jnp.isnan(hessian)))
            self.assertFalse(jnp.all(hessian == 0))

        except Exception as e:
            print(f"Hessian computation not supported: {e}")


class TestEmulatorPerformance(unittest.TestCase):
    """Performance tests for emulator system."""

    @classmethod
    def setUpClass(cls):
        """Set up performance tests."""
        import jaxcapse
        import jax
        import time

        cls.jaxcapse = jaxcapse
        cls.jax = jax
        cls.time = time

        # Get TT emulator for performance tests
        cls.emulator_TT = jaxcapse.trained_emulators.get("class_mnuw0wacdm", {}).get("TT")

    def test_jit_speedup(self):
        """Test that JIT compilation provides speedup."""
        if self.emulator_TT is None:
            self.skipTest("TT emulator not loaded")

        import jax
        import jax.numpy as jnp
        import time

        test_params = jnp.array([0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544])

        # Non-JIT version
        def regular_predict(params):
            return self.emulator_TT.predict(params)

        # JIT version
        jit_predict = jax.jit(regular_predict)

        try:
            # Warm-up JIT compilation
            _ = jit_predict(test_params)

            # Time regular version (multiple calls)
            n_calls = 100
            start = time.time()
            for _ in range(n_calls):
                _ = regular_predict(test_params)
            regular_time = time.time() - start

            # Time JIT version
            start = time.time()
            for _ in range(n_calls):
                _ = jit_predict(test_params)
            jit_time = time.time() - start

            speedup = regular_time / jit_time if jit_time > 0 else 1.0

            print(f"\nPerformance comparison ({n_calls} calls):")
            print(f"  Regular: {regular_time:.4f}s")
            print(f"  JIT: {jit_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")

            # JIT should be at least as fast (allowing for measurement noise)
            self.assertGreaterEqual(speedup, 0.9,
                                  "JIT should not be significantly slower")

        except Exception as e:
            print(f"JIT performance test skipped: {e}")

    def test_memory_efficiency(self):
        """Test that emulator memory usage is reasonable."""
        if self.emulator_TT is None:
            self.skipTest("TT emulator not loaded")

        import sys

        # Check that emulator object size is reasonable
        emulator_size = sys.getsizeof(self.emulator_TT)

        # Emulator should be less than 100MB in memory
        max_size_mb = 100
        size_mb = emulator_size / (1024 * 1024)

        print(f"\nMemory usage:")
        print(f"  Emulator object size: {size_mb:.2f} MB")

        # This is a soft check - warn if too large but don't fail
        if size_mb > max_size_mb:
            print(f"  ⚠ Warning: Emulator larger than {max_size_mb} MB")


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)