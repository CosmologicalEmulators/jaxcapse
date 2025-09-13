# Basic Usage Examples

## Simple Power Spectrum Computation

```python
import jaxcapse
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Load emulator
emulator = jaxcapse.load_emulator("trained_emu/TT/")

# Define cosmological parameters
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])

# Compute power spectrum
cl = emulator.get_Cl(params)

# Plot
ell = jnp.arange(2, len(cl) + 2)
plt.figure(figsize=(8, 5))
plt.plot(ell, ell * (ell + 1) * cl / (2 * jnp.pi))
plt.xlabel(r'$\ell$')
plt.ylabel(r'$D_\ell^{TT}$ [$\mu K^2$]')
plt.show()
```

## Parameter Variations

```python
# Vary one parameter
ns_values = jnp.linspace(0.92, 1.00, 5)
results = []

for ns in ns_values:
    params_varied = params.at[1].set(ns)
    cl = emulator.get_Cl(params_varied)
    results.append(cl)

# Plot variations
for i, (ns, cl) in enumerate(zip(ns_values, results)):
    plt.plot(ell, cl, label=f'ns={ns:.2f}')
plt.legend()
plt.show()
```