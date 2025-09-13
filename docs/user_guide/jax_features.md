# JAX Features

Leverage JAX's powerful transformations with JaxCapse.

## Automatic Differentiation

### Gradients

```python
import jax
import jax.numpy as jnp

# Define loss function
def loss(params, emulator):
    cl = emulator.get_Cl(params)
    return jnp.sum(cl**2)

# Compute gradient
grad_fn = jax.grad(loss, argnums=0)
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])
gradients = grad_fn(params, emulator)
```

### Jacobian

```python
# Forward-mode (efficient for many outputs)
jacobian = jax.jacfwd(emulator.get_Cl)(params)
print(f"Jacobian shape: {jacobian.shape}")  # (n_ell, n_params)

# Reverse-mode (efficient for many inputs)
jacobian_rev = jax.jacrev(emulator.get_Cl)(params)
```

## Vectorization

```python
# Vectorize single-input function
vmap_get_cl = jax.vmap(emulator.get_Cl)

# Process batch
batch = jnp.array([[3.1, 0.96, 67.0, 0.022, 0.12, 0.055],
                   [3.0, 0.97, 68.0, 0.023, 0.11, 0.060]])
cl_batch = vmap_get_cl(batch)
```

## JIT Compilation

```python
# JIT compile for speed
@jax.jit
def fast_compute(params):
    return emulator.get_Cl(params)

# First call compiles
cl = fast_compute(params)  # ~1ms

# Subsequent calls are fast
cl = fast_compute(params)  # ~50μs
```