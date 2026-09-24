# jaxcapse

**Fast, differentiable CMB power spectrum emulation in JAX**

## Overview

jaxcapse is a JAX implementation of the `Capse.jl` (CMB Angular Power Spectrum Emulator) neural network emulator for computing CMB power spectra. It provides:

- ⚡ **Fast inference**: Compute CMB power spectra in microseconds
- 🎯 **High accuracy**: Sub-percent precision across a wide parameter range
- 🔄 **Automatic differentiation**: Compute gradients/jacobians with respect to cosmological parameters
- 🚀 **Batch processing**: Efficiently process multiple parameter sets simultaneously
- 🔧 **JAX integration**: Full compatibility with JAX transformations (JIT, vmap, grad)

## Features

### Speed
jaxcapse computes CMB power spectra orders of magnitude faster than traditional Boltzmann solvers:

- Single evaluation: ~50 μs
- Gradient computation: ~400 μs
- Batch of 1000: ~10 ms

### Differentiability
Leverage JAX's automatic differentiation for:

- Parameter estimation
- Fisher matrix computation
- Sensitivity analysis
- Gradient-based optimization

### Supported Spectra
jaxcapse provides trained emulators for:

- **TT**: Temperature-Temperature
- **EE**: E-mode polarization
- **TE**: Temperature-E-mode cross-correlation
- **BB**: B-mode polarization
- **PP**: Lensing potential

## Quick Example

```python
import jaxcapse
import jax.numpy as jnp

# Access a bundled emulator (all five models are loaded on import)
emulator_TT = jaxcapse.trained_emulators["camb_mnuw0wacdm"]["TT"]

# Parameter order: [ln10As, ns, tau, H0, omega_b, omega_c, Mnu, w0, wa]
params = jnp.array([3.044, 0.965, 0.054, 67.4, 0.02237, 0.12, 0.06, -1.0, 0.0])
# The training domain additionally requires w0 + wa < -0.5.

# Compute power spectrum
cl_TT = emulator_TT.get_Cl(params)

# Compute jacobians
import jax
jacobian = jax.jacfwd(emulator_TT.get_Cl)(params)
```

## Installation

```bash
pip install jaxcapse
```

Or install from source:

```bash
git clone https://github.com/CosmologicalEmulators/jaxcapse.git
cd jaxcapse
pip install -e .
```

## Documentation

- **[API Reference](api.md)**: Complete API documentation
- **[Contributing](contributing.md)**: Contributing guidelines

## Requirements

- Python ≥ 3.10
- JAX ≥ 0.4.30
- Flax ≥ 0.10.0
- jaxace ≥ 0.7.0

## Citation

If you use jaxcapse in your research, please cite:

```bibtex
@article{Bonici2024Capse,
	author = {Bonici, Marco and Bianchini, Federico and Ruiz-Zapatero, Jaime},
	journal = {The Open Journal of Astrophysics},
	doi = {10.21105/astro.2307.14339},
	year = {2024},
	month = {jan 30},
	publisher = {Maynooth Academic Publishing},
	title = {Capse.jl: efficient and auto-differentiable {CMB} power spectra emulation},
	volume = {7},
}
```

## License

jaxcapse is released under the MIT License. See [LICENSE](https://github.com/CosmologicalEmulators/jaxcapse/blob/main/LICENSE) for details.

## Acknowledgments

jaxcapse builds on:

- [jaxace](https://github.com/CosmologicalEmulators/jaxace): JAX implementation of AbstractCosmologicalEmulators.jl
- [Capse.jl](https://github.com/CosmologicalEmulators/Capse.jl): Original Julia implementation
- [JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs
