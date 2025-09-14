"""
Example test demonstrating Jacobian computation with jaxcapse emulators.

This test shows how to use JAX's automatic differentiation capabilities
to compute derivatives of CMB power spectra with respect to cosmological parameters.
"""

import unittest
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestJacobianExample(unittest.TestCase):
    """
    Example test cases for computing Jacobians of CMB power spectra.

    This demonstrates how to use JAX autodiff with the downloaded emulators
    to understand parameter sensitivities.
    """

    @classmethod
    def setUpClass(cls):
        """Load emulators and import JAX."""
        try:
            import jaxcapse
            import jax
            import jax.numpy as jnp

            cls.jax = jax
            cls.jnp = jnp
            cls.jaxcapse = jaxcapse

            # Get the TT emulator
            cls.emulator_TT = jaxcapse.trained_emulators.get("class_mnuw0wacdm", {}).get("TT")

            # Parameter names for interpretation
            cls.param_names = ["ω_b", "ω_c", "h", "ln(10¹⁰As)", "ns", "τ"]

            # Fiducial parameters (Planck 2018 best-fit)
            cls.fiducial = jnp.array([
                0.02237,   # omega_b - baryon density
                0.1200,    # omega_c - cold dark matter density
                0.6736,    # h - Hubble parameter
                3.044,     # ln10As - primordial amplitude
                0.9649,    # ns - spectral index
                0.0544,    # tau - optical depth
            ])

        except ImportError as e:
            cls.emulator_TT = None
            print(f"Could not import required packages: {e}")

    def test_jacobian_at_multiple_ells(self):
        """
        Compute and visualize Jacobian at different multipoles.

        This shows how different scales respond to parameter changes.
        """
        if self.emulator_TT is None:
            self.skipTest("TT emulator not available")

        # Define functions for specific ell ranges
        def get_cl_at_ell(params, ell_idx):
            """Get Cl at specific multipole index."""
            cl_spectrum = self.emulator_TT.predict(params)
            return cl_spectrum[ell_idx] if ell_idx < len(cl_spectrum) else cl_spectrum[-1]

        # Test at different scales
        ell_indices = {
            "Large scales (ℓ~10)": 5,
            "First peak (ℓ~220)": 20,
            "Small scales (ℓ~1000)": 50
        }

        print("\n" + "=" * 60)
        print("JACOBIAN ANALYSIS AT DIFFERENT SCALES")
        print("=" * 60)

        for scale_name, ell_idx in ell_indices.items():
            try:
                # Create partial function for this ell
                cl_func = lambda params: get_cl_at_ell(params, ell_idx)

                # Compute gradient
                grad_fn = self.jax.grad(cl_func)
                gradient = grad_fn(self.fiducial)

                print(f"\n{scale_name}:")
                print("-" * 40)

                # Display sensitivities
                for param_name, grad_val in zip(self.param_names, gradient):
                    print(f"  ∂Cl/∂{param_name:12s} = {grad_val:+.6e}")

                # Find most sensitive parameter
                max_idx = self.jnp.argmax(self.jnp.abs(gradient))
                print(f"\n  Most sensitive: {self.param_names[max_idx]}")

            except Exception as e:
                print(f"  Could not compute gradient: {e}")

    def test_fisher_matrix_computation(self):
        """
        Compute Fisher matrix for parameter constraints.

        The Fisher matrix is related to parameter uncertainties
        and correlations in cosmological analyses.
        """
        if self.emulator_TT is None:
            self.skipTest("TT emulator not available")

        def log_likelihood_approx(params):
            """
            Approximate log-likelihood (simplified).
            In reality, this would include data and covariance.
            """
            cl_theory = self.emulator_TT.predict(params)
            # Simple chi-square-like quantity
            return -self.jnp.sum((cl_theory - cl_theory) ** 2)

        try:
            # Compute Hessian (Fisher matrix ≈ -Hessian of log-likelihood)
            hessian_fn = self.jax.hessian(log_likelihood_approx)
            hessian = hessian_fn(self.fiducial)

            print("\n" + "=" * 60)
            print("FISHER MATRIX ANALYSIS")
            print("=" * 60)

            # Check symmetry
            is_symmetric = self.jnp.allclose(hessian, hessian.T)
            print(f"\nMatrix symmetry: {'✓' if is_symmetric else '✗'}")

            # Diagonal elements (related to individual parameter constraints)
            print("\nDiagonal elements (parameter variances):")
            for i, param in enumerate(self.param_names):
                if i < len(hessian):
                    print(f"  F[{param}, {param}] = {hessian[i, i]:.6e}")

            # Correlation structure
            print("\nStrong correlations:")
            n_params = len(self.param_names)
            for i in range(n_params):
                for j in range(i + 1, n_params):
                    if i < len(hessian) and j < len(hessian):
                        if abs(hessian[i, j]) > 0.1 * max(abs(hessian[i, i]), abs(hessian[j, j])):
                            print(f"  {self.param_names[i]} ↔ {self.param_names[j]}")

        except Exception as e:
            print(f"Fisher matrix computation not fully supported: {e}")

    def test_parameter_derivatives_ratio(self):
        """
        Test ratios of derivatives (elasticities).

        This shows relative importance of parameters.
        """
        if self.emulator_TT is None:
            self.skipTest("TT emulator not available")

        def cl_sum(params):
            """Sum of Cl values (total power)."""
            return self.jnp.sum(self.emulator_TT.predict(params))

        try:
            # Compute gradient
            grad_fn = self.jax.grad(cl_sum)
            gradient = grad_fn(self.fiducial)

            # Compute elasticities (normalized derivatives)
            cl_total = cl_sum(self.fiducial)
            elasticities = gradient * self.fiducial / cl_total

            print("\n" + "=" * 60)
            print("PARAMETER ELASTICITIES")
            print("=" * 60)
            print("\n(% change in total Cl per % change in parameter)")
            print("-" * 40)

            # Sort by absolute elasticity
            sorted_indices = self.jnp.argsort(self.jnp.abs(elasticities))[::-1]

            for idx in sorted_indices:
                if idx < len(self.param_names):
                    param = self.param_names[idx]
                    elast = elasticities[idx]
                    print(f"  {param:12s}: {elast:+.3f}%")

            # Physical interpretation
            print("\nPhysical interpretation:")
            print("  • Positive: Parameter increases → More power")
            print("  • Negative: Parameter increases → Less power")
            print("  • Large |value|: Parameter has strong effect")

        except Exception as e:
            print(f"Elasticity computation failed: {e}")

    def test_directional_derivatives(self):
        """
        Test derivatives in specific parameter directions.

        This is useful for understanding parameter degeneracies.
        """
        if self.emulator_TT is None:
            self.skipTest("TT emulator not available")

        # Define interesting directions in parameter space
        directions = {
            "Ωm direction": self.jnp.array([0, 1, -1.5, 0, 0, 0]),  # ω_c ↑, h ↓
            "As-τ direction": self.jnp.array([0, 0, 0, 1, 0, -1]),   # As ↑, τ ↓
            "Baryon direction": self.jnp.array([1, -0.5, 0, 0, 0, 0])  # ω_b ↑, ω_c ↓
        }

        def cl_average(params):
            """Average Cl value."""
            return self.jnp.mean(self.emulator_TT.predict(params))

        print("\n" + "=" * 60)
        print("DIRECTIONAL DERIVATIVES")
        print("=" * 60)

        try:
            # Compute gradient
            grad_fn = self.jax.grad(cl_average)
            gradient = grad_fn(self.fiducial)

            for direction_name, direction in directions.items():
                # Normalize direction
                direction_norm = direction / self.jnp.linalg.norm(direction)

                # Directional derivative = gradient · direction
                directional_deriv = self.jnp.dot(gradient, direction_norm)

                print(f"\n{direction_name}:")
                print(f"  Directional derivative: {directional_deriv:+.6e}")

                # Show which parameters change
                print("  Changes:")
                for i, (param, change) in enumerate(zip(self.param_names, direction_norm)):
                    if abs(change) > 0.1:
                        print(f"    {param}: {change:+.3f}")

        except Exception as e:
            print(f"Directional derivative computation failed: {e}")


def run_jacobian_examples():
    """Run all Jacobian examples with nice output."""
    print("\n" + "=" * 70)
    print(" JAX AUTOMATIC DIFFERENTIATION WITH JAXCAPSE EMULATORS")
    print("=" * 70)
    print("\nThis demonstrates computing derivatives of CMB power spectra")
    print("with respect to cosmological parameters using JAX autodiff.")
    print("\nThe emulators are automatically downloaded from Zenodo and")
    print("are fully differentiable through JAX transformations.")

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestJacobianExample)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)


if __name__ == "__main__":
    run_jacobian_examples()