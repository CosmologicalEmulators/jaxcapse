# Quick Start

This guide will get you computing CMB power spectra with JaxCapse in minutes.

## Basic Usage

### 1. Import JaxCapse

```python
import jaxcapse
import jax.numpy as jnp
```

### 2. Load an Emulator

```python
# Load the temperature-temperature (TT) emulator
emulator_TT = jaxcapse.load_emulator("trained_emu/TT/")
```

### 3. Check Emulator Information

```python
# View emulator description
print(emulator_TT.emulator_description)
```

Output:
```python
{
    'author': 'Marco Bonici',
    'author_email': 'bonici.marco@gmail.com',
    'parameters': 'ln10As, ns, H0, ωb, ωc, τ',
    'miscellanea': 'Trained on CAMB high-precision predictions'
}
```

### 4. Define Cosmological Parameters

The parameters must be provided in the order specified by the emulator:

```python
# Parameters: [ln10As, ns, H0, ωb, ωc, τ]
params = jnp.array([
    3.1,    # ln(10^10 As)
    0.96,   # Spectral index
    67.0,   # Hubble constant [km/s/Mpc]
    0.022,  # Physical baryon density
    0.12,   # Physical CDM density
    0.055   # Optical depth to reionization
])
```

### 5. Compute Power Spectrum

```python
# Compute Cl values
cl_TT = emulator_TT.get_Cl(params)

print(f"Shape: {cl_TT.shape}")
print(f"First 5 values: {cl_TT[:5]}")
```

## Complete Example

Here's a complete example that loads all four emulators and computes all spectra:

```python
import jaxcapse
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Load all emulators
emulators = {
    'TT': jaxcapse.load_emulator("trained_emu/TT/"),
    'EE': jaxcapse.load_emulator("trained_emu/EE/"),
    'TE': jaxcapse.load_emulator("trained_emu/TE/"),
    'PP': jaxcapse.load_emulator("trained_emu/PP/")
}

# Define cosmological parameters
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])

# Compute all spectra
spectra = {}
for name, emulator in emulators.items():
    spectra[name] = emulator.get_Cl(params)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ell = jnp.arange(2, len(spectra['TT']) + 2)

for ax, (name, cl) in zip(axes.flat, spectra.items()):
    ax.plot(ell, cl)
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(f'$C_\\ell^{{{name}}}$')
    ax.set_title(f'{name} Spectrum')
    if name == 'PP':
        ax.set_yscale('log')

plt.tight_layout()
plt.show()
```

## Performance Timing

Check the speed of JaxCapse:

```python
import time

# Time single evaluation
start = time.perf_counter()
cl = emulator_TT.get_Cl(params)
end = time.perf_counter()
print(f"Single evaluation: {(end - start)*1e6:.1f} μs")

# Time after JIT compilation (second call)
start = time.perf_counter()
cl = emulator_TT.get_Cl(params)
end = time.perf_counter()
print(f"After JIT: {(end - start)*1e6:.1f} μs")
```

## Parameter Ranges

Each emulator is trained on specific parameter ranges. Check the training bounds:

```python
# View input parameter bounds
print("Parameter ranges:")
print(emulator_TT.in_MinMax)
```

The bounds are given as `[min, max]` for each parameter:
- ln10As: [2.5, 3.5]
- ns: [0.88, 1.05]
- H0: [40, 100]
- ωb: [0.019, 0.025]
- ωc: [0.08, 0.20]
- τ: [0.02, 0.12]

!!! warning "Stay Within Training Bounds"
    For accurate results, keep parameters within the training ranges. The emulator may produce unreliable outputs for parameters outside these bounds.

## Next Steps

- Learn about [computing gradients](../user_guide/jax_features.md#gradients)
- Explore [batch processing](../user_guide/batch.md)
- See the [detailed tutorial](tutorial.md)