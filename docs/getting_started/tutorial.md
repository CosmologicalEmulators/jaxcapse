# Tutorial

This tutorial provides a comprehensive introduction to JaxCapse, covering basic usage through advanced features.

## Part 1: Basic Power Spectrum Computation

### Setting Up

```python
import jaxcapse
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

# Configure JAX for 64-bit precision
jax.config.update('jax_enable_x64', True)
```

### Loading and Exploring Emulators

```python
# Load the TT emulator
emulator_TT = jaxcapse.load_emulator("trained_emu/TT/")

# Examine the emulator structure
print("Emulator attributes:")
print(f"- Input normalization shape: {emulator_TT.in_MinMax.shape}")
print(f"- Output normalization shape: {emulator_TT.out_MinMax.shape}")
print(f"- Has postprocessing: {callable(emulator_TT.postprocessing)}")
```

### Understanding Parameter Space

```python
# Get parameter information
desc = emulator_TT.emulator_description
print(f"Parameters: {desc['parameters']}")

# Check training bounds
bounds = emulator_TT.in_MinMax
param_names = ['ln10As', 'ns', 'H0', 'ωb', 'ωc', 'τ']

for i, name in enumerate(param_names):
    print(f"{name}: [{bounds[i,0]:.3f}, {bounds[i,1]:.3f}]")
```

### Computing Your First Spectrum

```python
# Define a fiducial cosmology
fiducial_params = jnp.array([
    3.05,   # ln10As
    0.965,  # ns
    67.36,  # H0
    0.0224, # ωb
    0.120,  # ωc
    0.054   # τ
])

# Compute the TT spectrum
cl_TT = emulator_TT.get_Cl(fiducial_params)

# Plot the result
ell = jnp.arange(2, len(cl_TT) + 2)
plt.figure(figsize=(8, 5))
plt.plot(ell, cl_TT)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{TT}$ [$\mu K^2$]')
plt.title('CMB Temperature Power Spectrum')
plt.grid(True, alpha=0.3)
plt.show()
```

## Part 2: Parameter Variations

### Exploring Parameter Dependencies

```python
# Create parameter variations
def vary_parameter(base_params, param_idx, values):
    """Vary one parameter while keeping others fixed."""
    results = []
    for val in values:
        params = base_params.copy()
        params = params.at[param_idx].set(val)
        results.append(emulator_TT.get_Cl(params))
    return jnp.array(results)

# Vary spectral index
ns_values = jnp.linspace(0.92, 1.00, 5)
cl_ns_varied = vary_parameter(fiducial_params, 1, ns_values)

# Plot variations
plt.figure(figsize=(10, 6))
for i, ns in enumerate(ns_values):
    plt.plot(ell, cl_ns_varied[i], label=f'ns = {ns:.3f}')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{TT}$ [$\mu K^2$]')
plt.legend()
plt.title('Effect of Spectral Index on TT Spectrum')
plt.show()
```

### Computing Ratios

```python
# Compute ratios relative to fiducial
plt.figure(figsize=(10, 6))
for i, ns in enumerate(ns_values):
    ratio = cl_ns_varied[i] / cl_TT
    plt.plot(ell, ratio, label=f'ns = {ns:.3f}')

plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell / C_\ell^{\rm fid}$')
plt.axhline(1, color='black', linestyle='--', alpha=0.5)
plt.legend()
plt.title('Relative Change in TT Spectrum')
plt.show()
```

## Part 3: Gradients and Derivatives

### Computing Jacobians

```python
# Compute Jacobian matrix (derivatives w.r.t. all parameters)
jacobian_fn = jax.jacfwd(emulator_TT.get_Cl)
jacobian = jacobian_fn(fiducial_params)

print(f"Jacobian shape: {jacobian.shape}")  # (n_ell, n_params)

# Plot derivatives for selected ℓ values
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
ell_samples = [10, 100, 500, 1000, 2000, 3000]

for ax, ell_idx in zip(axes.flat, ell_samples):
    derivatives = jacobian[ell_idx - 2, :]  # Adjust for ℓ starting at 2
    ax.bar(param_names, derivatives)
    ax.set_title(f'$\\partial C_{{\\ell={ell_idx}}}^{{TT}} / \\partial \\theta$')
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Fisher Matrix Computation

```python
def compute_fisher_matrix(emulator, params, ell_max=3000):
    """Compute Fisher information matrix."""
    # Get Jacobian
    jacobian = jax.jacfwd(emulator.get_Cl)(params)[:ell_max-1]
    
    # Assume cosmic variance limited
    cl = emulator.get_Cl(params)[:ell_max-1]
    ell = jnp.arange(2, ell_max+1)
    
    # Covariance (simplified - cosmic variance only)
    variance = 2 * cl**2 / (2*ell + 1)
    
    # Fisher matrix
    fisher = jnp.zeros((6, 6))
    for l in range(len(ell)):
        fisher += jnp.outer(jacobian[l], jacobian[l]) / variance[l]
    
    return fisher

# Compute Fisher matrix
fisher = compute_fisher_matrix(emulator_TT, fiducial_params)

# Plot correlation matrix
correlation = jnp.zeros_like(fisher)
for i in range(6):
    for j in range(6):
        correlation = correlation.at[i,j].set(
            fisher[i,j] / jnp.sqrt(fisher[i,i] * fisher[j,j])
        )

plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(6), param_names, rotation=45)
plt.yticks(range(6), param_names)
plt.title('Parameter Correlation Matrix')
plt.show()
```

## Part 4: Batch Processing

### Processing Multiple Cosmologies

```python
# Generate a batch of cosmologies
n_samples = 100
key = jax.random.PRNGKey(42)

# Random variations around fiducial
variations = jax.random.normal(key, (n_samples, 6)) * 0.01
batch_params = fiducial_params + variations

# Ensure within bounds
batch_params = jnp.clip(batch_params, 
                        emulator_TT.in_MinMax[:, 0],
                        emulator_TT.in_MinMax[:, 1])

# Compute all spectra
cl_batch = emulator_TT.get_Cl_batch(batch_params)
print(f"Batch output shape: {cl_batch.shape}")

# Plot statistics
mean_cl = jnp.mean(cl_batch, axis=0)
std_cl = jnp.std(cl_batch, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(ell, mean_cl, 'b-', label='Mean')
plt.fill_between(ell, mean_cl - std_cl, mean_cl + std_cl, 
                 alpha=0.3, label='±1σ')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$C_\ell^{TT}$ [$\mu K^2$]')
plt.legend()
plt.title(f'Statistics from {n_samples} Cosmologies')
plt.show()
```

### Performance Comparison

```python
import time

# Time batch processing
start = time.perf_counter()
cl_batch = emulator_TT.get_Cl_batch(batch_params)
batch_time = time.perf_counter() - start

# Time individual processing
start = time.perf_counter()
cl_individual = jnp.array([emulator_TT.get_Cl(p) for p in batch_params])
individual_time = time.perf_counter() - start

print(f"Batch processing: {batch_time*1000:.1f} ms")
print(f"Individual processing: {individual_time*1000:.1f} ms")
print(f"Speedup: {individual_time/batch_time:.1f}x")
```

## Part 5: All Spectra Together

### Loading All Emulators

```python
# Load all available emulators
emulator_names = ['TT', 'EE', 'TE', 'PP']
emulators = {}

for name in emulator_names:
    path = f"trained_emu/{name}/"
    try:
        emulators[name] = jaxcapse.load_emulator(path)
        print(f"✓ Loaded {name} emulator")
    except:
        print(f"✗ Could not load {name} emulator")
```

### Computing Cross-Correlations

```python
# Compute all spectra
all_spectra = {name: em.get_Cl(fiducial_params) 
               for name, em in emulators.items()}

# Create comprehensive plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plot_config = {
    'TT': {'ax': axes[0,0], 'ylabel': r'$\ell(\ell+1)C_\ell^{TT}/2\pi$ [$\mu K^2$]'},
    'EE': {'ax': axes[0,1], 'ylabel': r'$\ell(\ell+1)C_\ell^{EE}/2\pi$ [$\mu K^2$]'},
    'TE': {'ax': axes[1,0], 'ylabel': r'$\ell(\ell+1)C_\ell^{TE}/2\pi$ [$\mu K^2$]'},
    'PP': {'ax': axes[1,1], 'ylabel': r'$\ell^2(\ell+1)^2C_\ell^{\phi\phi}/2\pi$'}
}

for name, cl in all_spectra.items():
    if name in plot_config:
        ax = plot_config[name]['ax']
        ell = jnp.arange(2, len(cl) + 2)
        
        # Apply appropriate ℓ scaling
        if name == 'PP':
            scaling = ell**2 * (ell + 1)**2 / (2*jnp.pi)
            ax.semilogy(ell, cl * scaling)
        else:
            scaling = ell * (ell + 1) / (2*jnp.pi)
            ax.plot(ell, cl * scaling)
        
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel(plot_config[name]['ylabel'])
        ax.set_title(f'{name} Spectrum')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Part 6: Advanced JAX Features

### Using vmap for Parameter Studies

```python
# Define a parameter grid
As_range = jnp.linspace(2.8, 3.3, 10)
ns_range = jnp.linspace(0.92, 1.00, 10)

# Create 2D grid
As_grid, ns_grid = jnp.meshgrid(As_range, ns_range)

# Prepare parameters (flatten grid)
grid_params = []
for i in range(10):
    for j in range(10):
        params = fiducial_params.copy()
        params = params.at[0].set(As_grid[i, j])
        params = params.at[1].set(ns_grid[i, j])
        grid_params.append(params)

grid_params = jnp.array(grid_params)

# Use vmap for efficient computation
vmap_get_cl = jax.vmap(emulator_TT.get_Cl)
cl_grid = vmap_get_cl(grid_params)

# Extract amplitude at specific ℓ
ell_target = 1000
amplitude_grid = cl_grid[:, ell_target-2].reshape(10, 10)

# Plot parameter dependence
plt.figure(figsize=(10, 8))
plt.contourf(As_range, ns_range, amplitude_grid, levels=20, cmap='viridis')
plt.colorbar(label=f'$C_{{{ell_target}}}^{{TT}}$ [$\\mu K^2$]')
plt.xlabel('ln10As')
plt.ylabel('ns')
plt.title(f'TT Spectrum Amplitude at $\\ell = {ell_target}$')
plt.show()
```

### Optimization Example

```python
from scipy.optimize import minimize

# Define a mock "observed" spectrum
true_params = jnp.array([3.08, 0.968, 68.0, 0.0223, 0.118, 0.056])
observed_cl = emulator_TT.get_Cl(true_params)

# Add noise
key = jax.random.PRNGKey(123)
noise = jax.random.normal(key, observed_cl.shape) * 10
observed_cl_noisy = observed_cl + noise

# Define chi-squared function
def chi2(params):
    params_jax = jnp.array(params)
    theory_cl = emulator_TT.get_Cl(params_jax)
    return float(jnp.sum((theory_cl - observed_cl_noisy)**2 / (2 * observed_cl)))

# Optimize
result = minimize(chi2, fiducial_params, method='L-BFGS-B',
                 bounds=[(b[0], b[1]) for b in emulator_TT.in_MinMax])

print("True parameters:     ", true_params)
print("Recovered parameters:", result.x)
print("Chi-squared:        ", result.fun)
```

## Summary

You've learned how to:

1. Load and use JaxCapse emulators
2. Compute CMB power spectra for different cosmologies
3. Calculate gradients and Fisher matrices
4. Process batches of parameters efficiently
5. Use advanced JAX features like vmap
6. Perform parameter optimization

## Next Steps

- Explore the [API Reference](../api/jaxcapse.md) for detailed documentation
- See [Examples](../examples/basic.md) for more use cases
- Learn about [JAX Features](../user_guide/jax_features.md) in depth