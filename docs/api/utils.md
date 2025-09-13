# Utilities

Utility functions used by JaxCapse.

## Normalization Functions

JaxCapse uses normalization functions from jaxace:

### maximin

Normalizes data to [0, 1] range using min-max scaling.

```python
from jaxace import maximin

normalized = maximin(data, minmax_array)
```

### inv_maximin

Inverse normalization to recover original scale.

```python
from jaxace import inv_maximin

original = inv_maximin(normalized_data, minmax_array)
```

## File Structure

### Required Files for Emulator

Each trained emulator directory must contain:

| File | Description |
|------|-------------|
| `nn_setup.json` | Neural network architecture specification |
| `weights.npy` | Trained neural network weights |
| `inminmax.npy` | Input normalization parameters |
| `outminmax.npy` | Output normalization parameters |
| `postprocessing.py` | Spectrum-specific postprocessing function |

### nn_setup.json Structure

```json
{
    "n_input_features": 6,
    "n_output_features": 4999,
    "n_hidden_layers": 3,
    "layers": {
        "layer_1": {
            "n_neurons": 256,
            "activation_function": "tanh"
        },
        "layer_2": {
            "n_neurons": 256,
            "activation_function": "tanh"
        },
        "layer_3": {
            "n_neurons": 128,
            "activation_function": "relu"
        }
    },
    "emulator_description": {
        "author": "Marco Bonici",
        "author_email": "bonici.marco@gmail.com",
        "parameters": "ln10As, ns, H0, ωb, ωc, τ",
        "miscellanea": "Trained on CAMB predictions"
    }
}
```

### Postprocessing Function

The postprocessing function must have the signature:

```python
def postprocessing(input_params, output):
    """
    Apply spectrum-specific postprocessing.
    
    Args:
        input_params: Cosmological parameters (JAX array)
        output: Neural network output (JAX array)
    
    Returns:
        Processed power spectrum values
    """
    import jax.numpy as jnp
    return output * jnp.exp(input_params[0] - 3.0)
```

## JAX Configuration

### Precision Settings

JaxCapse uses 64-bit precision by default:

```python
import jax
jax.config.update('jax_enable_x64', True)
```

### Device Selection

Check available devices:

```python
import jax
print(jax.devices())  # Lists available devices
```

Force CPU usage:

```python
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
```

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Missing emulator files | Ensure all required files are in the directory |
| `AttributeError: 'list' object has no attribute 'ndim'` | Passing list instead of array | Convert to JAX array: `jnp.array(params)` |
| `ValueError: shapes do not match` | Wrong number of parameters | Check parameter order and count |
| `json.JSONDecodeError` | Corrupted nn_setup.json | Verify JSON file is valid |

## Performance Tips

### JIT Compilation

- First call compiles the function (slower)
- Subsequent calls use compiled version (faster)
- Compilation is shape-specific

### Memory Management

For large batches:
```python
# Process in chunks if needed
chunk_size = 1000
results = []
for i in range(0, len(large_batch), chunk_size):
    chunk = large_batch[i:i+chunk_size]
    results.append(emulator.get_Cl_batch(chunk))
result = jnp.concatenate(results)
```

### Gradient Computation

For many outputs, use forward-mode differentiation:
```python
# Efficient for n_outputs >> n_inputs
jacobian = jax.jacfwd(emulator.get_Cl)(params)
```

For many inputs, use reverse-mode:
```python
# Efficient for n_inputs >> n_outputs
jacobian = jax.jacrev(emulator.get_Cl)(params)
```