# Batch Processing Examples

## Processing Multiple Cosmologies

```python
import jaxcapse
import jax.numpy as jnp
import jax

# Load emulator
emulator = jaxcapse.load_emulator("trained_emu/TT/")

# Generate random cosmologies
key = jax.random.PRNGKey(42)
n_samples = 100

# Random variations around fiducial
fiducial = jnp.array([3.05, 0.965, 67.36, 0.0224, 0.120, 0.054])
variations = jax.random.normal(key, (n_samples, 6)) * 0.01
batch_params = fiducial + variations

# Process batch
cl_batch = emulator.get_Cl_batch(batch_params)

# Compute statistics
mean_cl = jnp.mean(cl_batch, axis=0)
std_cl = jnp.std(cl_batch, axis=0)

print(f"Processed {n_samples} cosmologies")
print(f"Mean Cl shape: {mean_cl.shape}")
```

## MCMC Sampling

```python
def log_likelihood(params, data, emulator):
    """Compute log likelihood for MCMC."""
    cl_theory = emulator.get_Cl(params)
    chi2 = jnp.sum((cl_theory - data['cl'])**2 / data['variance'])
    return -0.5 * chi2

# Use in MCMC sampler
# (example with emcee or other samplers)
```