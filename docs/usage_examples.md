# Usage Examples

This guide demonstrates how to use jaxcapse to compute and visualize CMB power spectra and their derivatives.

## Quick Start

```python
import jaxcapse
import jax.numpy as jnp
import matplotlib.pyplot as plt

# The emulators are automatically loaded when you import jaxcapse
# Access them via the trained_emulators dictionary
emulators = jaxcapse.trained_emulators["class_mnuw0wacdm"]
```

## Computing Power Spectra

### Basic Usage

```python
# Define cosmological parameters
# Order: [omega_b, omega_c, h, ln10As, ns, tau]
params = jnp.array([
    0.02237,   # Baryon density
    0.1200,    # CDM density
    0.6736,    # Hubble parameter
    3.044,     # Log primordial amplitude
    0.9649,    # Spectral index
    0.0544     # Optical depth
])

# Compute all power spectra
cl_tt = emulators["TT"].predict(params)  # Temperature
cl_ee = emulators["EE"].predict(params)  # E-mode polarization
cl_te = emulators["TE"].predict(params)  # Temperature-polarization cross
cl_pp = emulators["PP"].predict(params)  # Lensing potential
```

### Plotting All Spectra

```python
import matplotlib.pyplot as plt
import numpy as np

# Create multipole array (adjust based on your emulator output)
n_ells = len(cl_tt)
ell = np.arange(2, n_ells + 2)

# Create figure with subplots for all spectra
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot TT spectrum
axes[0, 0].loglog(ell, ell * (ell + 1) * cl_tt / (2 * np.pi))
axes[0, 0].set_xlabel(r'$\ell$')
axes[0, 0].set_ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu K^2$]')
axes[0, 0].set_title('Temperature Power Spectrum')
axes[0, 0].grid(True, alpha=0.3)

# Plot EE spectrum
axes[0, 1].loglog(ell, ell * (ell + 1) * cl_ee / (2 * np.pi))
axes[0, 1].set_xlabel(r'$\ell$')
axes[0, 1].set_ylabel(r'$\ell(\ell+1)C_\ell^{EE}/2\pi$ [$\mu K^2$]')
axes[0, 1].set_title('E-mode Polarization Spectrum')
axes[0, 1].grid(True, alpha=0.3)

# Plot TE spectrum (can be negative, use semilogy with abs)
cl_te_plot = ell * (ell + 1) * np.abs(cl_te) / (2 * np.pi)
axes[1, 0].loglog(ell, cl_te_plot)
axes[1, 0].set_xlabel(r'$\ell$')
axes[1, 0].set_ylabel(r'$|\ell(\ell+1)C_\ell^{TE}/2\pi|$ [$\mu K^2$]')
axes[1, 0].set_title('Temperature-Polarization Cross Spectrum')
axes[1, 0].grid(True, alpha=0.3)

# Plot PP spectrum (lensing potential)
axes[1, 1].loglog(ell, ell * (ell + 1) * cl_pp)
axes[1, 1].set_xlabel(r'$\ell$')
axes[1, 1].set_ylabel(r'$\ell(\ell+1)C_\ell^{\phi\phi}$')
axes[1, 1].set_title('Lensing Potential Spectrum')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cmb_spectra.png', dpi=150, bbox_inches='tight')
plt.show()
```

![CMB Power Spectra](images/cmb_spectra.png)

## Computing Jacobians with JAX

One of the powerful features of jaxcapse is that the emulators are fully differentiable using JAX's automatic differentiation.

### Basic Jacobian Computation

```python
import jax
import jax.numpy as jnp

# Define a function that computes TT spectrum from parameters
def compute_cl_tt(params):
    """Compute TT power spectrum for given parameters."""
    return emulators["TT"].predict(params)

# Compute Jacobian using JAX autodiff
jacobian_fn = jax.jacobian(compute_cl_tt)
jacobian = jacobian_fn(params)

print(f"Jacobian shape: {jacobian.shape}")
# Output: (n_ell, n_params) - derivative of each Cl with respect to each parameter
```

### Visualizing Parameter Sensitivities

```python
# Parameter names for labeling
param_names = [r'$\omega_b$', r'$\omega_c$', r'$h$',
               r'$\ln(10^{10}A_s)$', r'$n_s$', r'$\tau$']

# Create figure showing Jacobian for each parameter
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (ax, name) in enumerate(zip(axes, param_names)):
    # Plot derivative of Cl_TT with respect to parameter i
    ax.semilogx(ell, jacobian[:, i])
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(rf'$\partial C_\ell^{{TT}}/\partial {name}$')
    ax.set_title(f'Sensitivity to {name}')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

plt.suptitle('CMB TT Power Spectrum Jacobian', fontsize=16)
plt.tight_layout()
plt.savefig('jacobian_tt.png', dpi=150, bbox_inches='tight')
plt.show()
```

![TT Jacobian](images/jacobian_tt.png)

### Normalized Derivatives (Elasticities)

To understand relative parameter importance:

```python
# Compute elasticities (percent change in Cl per percent change in parameter)
elasticities = jacobian * params[None, :] / cl_tt[:, None]

# Plot elasticities at different scales
fig, ax = plt.subplots(figsize=(10, 6))

for i, name in enumerate(param_names):
    ax.semilogx(ell, elasticities[:, i], label=name, linewidth=2)

ax.set_xlabel(r'$\ell$')
ax.set_ylabel('Elasticity [%/%]')
ax.set_title('Parameter Elasticities for TT Spectrum')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('elasticities_tt.png', dpi=150, bbox_inches='tight')
plt.show()
```

![TT Elasticities](images/elasticities_tt.png)

## Advanced Usage

### Computing Fisher Matrix

```python
# Define log-likelihood function (simplified)
def log_likelihood(params):
    """Simplified log-likelihood for Fisher matrix computation."""
    cl_theory = emulators["TT"].predict(params)
    # In practice, this would include data and covariance
    return -jnp.sum((cl_theory - cl_theory) ** 2)

# Compute Hessian (Fisher matrix ≈ -Hessian of log-likelihood)
hessian_fn = jax.hessian(log_likelihood)
fisher_matrix = -hessian_fn(params)

print(f"Fisher matrix shape: {fisher_matrix.shape}")
```

### Batch Predictions with vmap

Process multiple parameter sets efficiently:

```python
# Generate batch of parameter variations
n_samples = 100
param_variations = jax.random.normal(
    jax.random.PRNGKey(42),
    shape=(n_samples, 6)
) * 0.01 + params

# Vectorize prediction over batch
batch_predict = jax.vmap(emulators["TT"].predict)
batch_spectra = batch_predict(param_variations)

print(f"Batch output shape: {batch_spectra.shape}")
# Output: (n_samples, n_ell)

# Plot mean and standard deviation
mean_spectrum = jnp.mean(batch_spectra, axis=0)
std_spectrum = jnp.std(batch_spectra, axis=0)

plt.figure(figsize=(10, 6))
plt.loglog(ell, ell * (ell + 1) * mean_spectrum / (2 * np.pi),
           'b-', label='Mean')
plt.fill_between(ell,
                  ell * (ell + 1) * (mean_spectrum - std_spectrum) / (2 * np.pi),
                  ell * (ell + 1) * (mean_spectrum + std_spectrum) / (2 * np.pi),
                  alpha=0.3, label='±1σ')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu K^2$]')
plt.title('Parameter Variation Effects on TT Spectrum')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### JIT Compilation for Performance

```python
# JIT compile the prediction function for better performance
@jax.jit
def fast_predict(params):
    return emulators["TT"].predict(params)

# First call compiles the function
cl_tt_fast = fast_predict(params)

# Subsequent calls are much faster
cl_tt_fast = fast_predict(params)
```

## Physical Interpretation

### Understanding the Jacobians

The sign and magnitude of Jacobian elements have physical meaning:

- **Positive derivatives**: Parameter increase → More power at that scale
- **Negative derivatives**: Parameter increase → Less power at that scale
- **Large magnitude**: Strong sensitivity to parameter
- **Scale dependence**: Different parameters affect different scales

### Key Parameter Effects

1. **$\omega_b$ (Baryon density)**: Affects acoustic peak heights and positions
2. **$\omega_c$ (CDM density)**: Changes overall amplitude and peak structure
3. **$h$ (Hubble)**: Shifts angular scale of features
4. **$\ln(10^{10}A_s)$ (Primordial amplitude)**: Overall normalization
5. **$n_s$ (Spectral index)**: Tilt of primordial spectrum
6. **$\tau$ (Optical depth)**: Suppression at small scales

## Complete Example Script

Here's a complete script that generates all the plots:

```python
import jaxcapse
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Set up parameters
params = jnp.array([0.02237, 0.1200, 0.6736, 3.044, 0.9649, 0.0544])
param_names = [r'$\omega_b$', r'$\omega_c$', r'$h$',
               r'$\ln(10^{10}A_s)$', r'$n_s$', r'$\tau$']

# Get emulators
emulators = jaxcapse.trained_emulators["class_mnuw0wacdm"]

# Compute all spectra
cl_tt = emulators["TT"].predict(params)
cl_ee = emulators["EE"].predict(params)
cl_te = emulators["TE"].predict(params)
cl_pp = emulators["PP"].predict(params)

# Create ell array
n_ells = len(cl_tt)
ell = np.arange(2, n_ells + 2)

# Plot all spectra
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
# ... (plotting code as shown above)

# Compute and plot Jacobian
jacobian_fn = jax.jacobian(emulators["TT"].predict)
jacobian = jacobian_fn(params)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# ... (Jacobian plotting code as shown above)

print("All plots generated successfully!")
```

## Tips and Best Practices

1. **Parameter ranges**: Keep parameters within training ranges for accurate predictions
2. **JIT compilation**: Use `@jax.jit` for repeated evaluations
3. **Batch processing**: Use `vmap` for multiple parameter sets
4. **Gradient checks**: Verify Jacobians have expected physical behavior
5. **Memory management**: Emulators are loaded once at import