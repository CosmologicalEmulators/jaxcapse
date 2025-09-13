# MLP Class

The main emulator class for computing CMB power spectra.

## Class Definition

::: jaxcapse.jaxcapse.MLP
    options:
      show_source: true
      members:
        - __init__
        - get_Cl
        - get_Cl_batch

## Attributes

### emulator
The underlying jaxace FlaxEmulator instance containing the neural network model.

### in_MinMax
JAX array of shape `(n_params, 2)` containing the minimum and maximum values for input normalization.

### out_MinMax
JAX array of shape `(n_outputs, 2)` containing the minimum and maximum values for output normalization.

### postprocessing
Callable function that applies spectrum-specific postprocessing to the neural network output.

### emulator_description
Dictionary containing metadata about the emulator:
- `author`: Creator of the emulator
- `author_email`: Contact email
- `parameters`: Comma-separated list of parameter names
- `miscellanea`: Additional information about training

## Methods

### get_Cl

Compute CMB power spectrum for a single set of cosmological parameters.

**Parameters:**
- `input_data` (jnp.ndarray): Cosmological parameters as JAX array of shape (n_params,)

**Returns:**
- `jnp.ndarray`: Power spectrum Cl values

**Example:**
```python
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])
cl = emulator.get_Cl(params)
```

### get_Cl_batch

Compute CMB power spectra for multiple sets of cosmological parameters.

**Parameters:**
- `input_batch` (np.ndarray): Array of cosmological parameters, shape (n_samples, n_params)

**Returns:**
- `np.ndarray`: Array of power spectra, shape (n_samples, n_cls)

**Example:**
```python
batch_params = jnp.array([
    [3.1, 0.96, 67.0, 0.022, 0.12, 0.055],
    [3.0, 0.97, 68.0, 0.023, 0.11, 0.060]
])
cl_batch = emulator.get_Cl_batch(batch_params)
```

## JAX Integration

The MLP class is fully compatible with JAX transformations:

### JIT Compilation
The `get_Cl` method is JIT-compiled by default for optimal performance:
```python
# First call includes compilation
cl1 = emulator.get_Cl(params)  # ~1ms

# Subsequent calls use compiled version
cl2 = emulator.get_Cl(params)  # ~50μs
```

### Automatic Differentiation
```python
import jax

# Compute gradients
grad_fn = jax.grad(lambda p: jnp.sum(emulator.get_Cl(p)))
gradients = grad_fn(params)

# Compute Jacobian
jacobian = jax.jacfwd(emulator.get_Cl)(params)
```

### Vectorization
```python
# Vectorize for batch processing
vmap_get_cl = jax.vmap(emulator.get_Cl)
cl_batch = vmap_get_cl(batch_params)
```

## Notes

- Input parameters must be JAX arrays (no automatic conversion from lists)
- The emulator uses jaxace infrastructure for neural network operations
- Normalization is handled internally using jaxace's maximin functions
- JIT compilation provides significant speedup after the first call