# jaxcapse

Main module for JaxCapse CMB power spectrum emulation.

## Functions

### load_emulator

::: jaxcapse.jaxcapse.load_emulator
    options:
      show_source: true
      
### load_preprocessing

::: jaxcapse.jaxcapse.load_preprocessing
    options:
      show_source: true

## Example Usage

```python
import jaxcapse
import jax.numpy as jnp

# Load an emulator
emulator = jaxcapse.load_emulator("trained_emu/TT/")

# Define cosmological parameters
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])

# Compute power spectrum
cl = emulator.get_Cl(params)
```

## Module Structure

The `jaxcapse` module provides:

- **`load_emulator`**: Main function to load trained neural network emulators
- **`load_preprocessing`**: Utility to load postprocessing functions
- **`MLP`**: Main emulator class (see [MLP API](mlp.md))

## Dependencies

JaxCapse depends on:

- `jax`: For automatic differentiation and JIT compilation
- `jaxace`: For neural network infrastructure
- `flax`: For neural network models
- `numpy`: For array operations

## Configuration

JaxCapse uses 64-bit precision by default:

```python
import jax
jax.config.update('jax_enable_x64', True)
```