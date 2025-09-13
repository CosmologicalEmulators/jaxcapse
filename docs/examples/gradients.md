# Gradient Examples

## Computing Derivatives

```python
import jax
import jax.numpy as jnp
import jaxcapse

# Load emulator
emulator = jaxcapse.load_emulator("trained_emu/TT/")
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])

# Gradient of sum of Cl
grad_fn = jax.grad(lambda p: jnp.sum(emulator.get_Cl(p)))
gradients = grad_fn(params)

print("Parameter gradients:", gradients)
```

## Fisher Matrix

```python
# Compute Jacobian
jacobian = jax.jacfwd(emulator.get_Cl)(params)

# Compute Fisher matrix (simplified)
cl = emulator.get_Cl(params)
ell = jnp.arange(2, len(cl) + 2)
variance = 2 * cl**2 / (2*ell + 1)

fisher = jnp.zeros((6, 6))
for l in range(len(cl)):
    fisher += jnp.outer(jacobian[l], jacobian[l]) / variance[l]

print("Fisher matrix shape:", fisher.shape)
```