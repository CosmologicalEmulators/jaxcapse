# Computing Power Spectra

This guide covers computing CMB power spectra with JaxCapse.

## Basic Computation

### Single Evaluation

```python
import jaxcapse
import jax.numpy as jnp

# Load emulator
emulator = jaxcapse.load_emulator("trained_emu/TT/")

# Define parameters [ln10As, ns, H0, ωb, ωc, τ]
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])

# Compute spectrum
cl = emulator.get_Cl(params)
```

### Multiple Spectra

```python
# Load all emulators
emulators = {
    'TT': jaxcapse.load_emulator("trained_emu/TT/"),
    'EE': jaxcapse.load_emulator("trained_emu/EE/"),
    'TE': jaxcapse.load_emulator("trained_emu/TE/"),
    'PP': jaxcapse.load_emulator("trained_emu/PP/")
}

# Compute all spectra
spectra = {name: em.get_Cl(params) 
           for name, em in emulators.items()}
```

## Parameter Input

### Parameter Order

Parameters must be provided in this exact order:

1. **ln10As**: Natural log of 10^10 * As
2. **ns**: Scalar spectral index
3. **H0**: Hubble constant [km/s/Mpc]
4. **ωb**: Physical baryon density (Ωb * h²)
5. **ωc**: Physical CDM density (Ωc * h²)
6. **τ**: Optical depth to reionization

### Parameter Validation

```python
def validate_parameters(params, emulator):
    """Validate parameters are within bounds."""
    bounds = emulator.in_MinMax
    
    for i, (val, (min_val, max_val)) in enumerate(zip(params, bounds)):
        if not (min_val <= val <= max_val):
            param_names = ['ln10As', 'ns', 'H0', 'ωb', 'ωc', 'τ']
            raise ValueError(
                f"Parameter {param_names[i]} = {val} "
                f"outside bounds [{min_val}, {max_val}]"
            )
    
    return True

# Validate before computing
validate_parameters(params, emulator)
cl = emulator.get_Cl(params)
```

## Output Format

### Understanding Output

```python
# Compute spectrum
cl = emulator.get_Cl(params)

# Output properties
print(f"Shape: {cl.shape}")           # Number of ℓ values
print(f"Data type: {cl.dtype}")       # float64 by default
print(f"Min value: {cl.min():.2e}")   # Minimum Cl
print(f"Max value: {cl.max():.2e}")   # Maximum Cl

# Get ℓ values
ell = jnp.arange(2, len(cl) + 2)      # ℓ starts at 2
```

### Physical Units

Different spectra have different units:

| Spectrum | Units | Description |
|----------|-------|-------------|
| TT | μK² | Temperature fluctuations squared |
| EE | μK² | E-mode polarization squared |
| TE | μK² | Temperature-polarization cross |
| PP | dimensionless | Lensing potential |

### Applying ℓ Factors

```python
# Standard CMB plotting convention
ell = jnp.arange(2, len(cl) + 2)

# D_ℓ = ℓ(ℓ+1)C_ℓ/2π
dl = ell * (ell + 1) * cl / (2 * jnp.pi)

# For lensing potential
if spectrum_type == 'PP':
    # [ℓ(ℓ+1)]²C_ℓ^φφ/2π
    dl = (ell * (ell + 1))**2 * cl / (2 * jnp.pi)
```

## Performance Optimization

### JIT Compilation

```python
# First call includes compilation
start = time.time()
cl1 = emulator.get_Cl(params)
print(f"First call: {(time.time() - start)*1000:.1f} ms")

# Subsequent calls are faster
start = time.time()
cl2 = emulator.get_Cl(params)
print(f"Second call: {(time.time() - start)*1000:.1f} ms")
```

### Avoiding Recompilation

```python
# Recompilation happens with different shapes
params_6d = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])
params_5d = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12])  # Wrong!

# This causes recompilation (and error)
# cl = emulator.get_Cl(params_5d)  # Don't do this
```

## Advanced Computation

### Custom Postprocessing

```python
def compute_with_custom_processing(emulator, params, custom_fn):
    """Apply custom postprocessing to spectrum."""
    # Get raw spectrum
    cl = emulator.get_Cl(params)
    
    # Apply custom processing
    processed = custom_fn(cl)
    
    return processed

# Example: Apply smoothing
def smooth_spectrum(cl, window=10):
    """Apply running mean smoothing."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(cl, window, mode='nearest')

cl_smooth = compute_with_custom_processing(
    emulator, params, lambda cl: smooth_spectrum(cl, 20)
)
```

### Combining Spectra

```python
def compute_chi_squared(params, observed_data):
    """Compute χ² for parameter set."""
    # Compute theory
    cl_theory = emulator.get_Cl(params)
    
    # Compute χ²
    residuals = cl_theory - observed_data['cl']
    chi2 = jnp.sum(residuals**2 / observed_data['variance'])
    
    return chi2

# Use in optimization
from scipy.optimize import minimize

result = minimize(
    lambda p: compute_chi_squared(jnp.array(p), data),
    initial_params,
    bounds=emulator.in_MinMax
)
```

### Spectrum Ratios

```python
def compute_ratio_spectrum(params1, params2, emulator):
    """Compute ratio of two spectra."""
    cl1 = emulator.get_Cl(params1)
    cl2 = emulator.get_Cl(params2)
    
    # Avoid division by zero
    ratio = jnp.where(cl2 > 0, cl1 / cl2, 1.0)
    
    return ratio

# Compare cosmologies
fiducial = jnp.array([3.05, 0.965, 67.36, 0.0224, 0.120, 0.054])
modified = jnp.array([3.10, 0.965, 67.36, 0.0224, 0.120, 0.054])

ratio = compute_ratio_spectrum(modified, fiducial, emulator)
```

## Error Handling

### Common Issues

```python
def safe_compute(emulator, params):
    """Compute spectrum with error handling."""
    try:
        # Validate input
        if not isinstance(params, jnp.ndarray):
            params = jnp.array(params)
        
        if params.shape != (6,):
            raise ValueError(f"Expected 6 parameters, got {len(params)}")
        
        # Check for NaN/Inf
        if not jnp.all(jnp.isfinite(params)):
            raise ValueError("Parameters contain NaN or Inf")
        
        # Compute
        cl = emulator.get_Cl(params)
        
        # Validate output
        if not jnp.all(jnp.isfinite(cl)):
            raise RuntimeError("Output contains NaN or Inf")
        
        return cl
        
    except Exception as e:
        print(f"Computation failed: {e}")
        return None
```

## Next Steps

- [Batch Processing](batch.md): Process multiple cosmologies
- [JAX Features](jax_features.md): Gradients and transformations
- [Examples](../examples/basic.md): Complete examples