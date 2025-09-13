# JaxCapse

<div align="center">

[![Tests](https://github.com/CosmologicalEmulators/jaxcapse/actions/workflows/tests.yml/badge.svg)](https://github.com/CosmologicalEmulators/jaxcapse/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/jaxcapse/branch/main/graph/badge.svg)](https://codecov.io/gh/CosmologicalEmulators/jaxcapse)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Fast, differentiable CMB power spectrum emulation in JAX*

</div>

## Overview

JaxCapse is a JAX implementation of the CAPSE (Cosmological Autoencoder for Power Spectrum Emulation) neural network emulator for computing CMB (Cosmic Microwave Background) power spectra. It provides:

- ⚡ **Fast inference**: Compute CMB power spectra in microseconds
- 🎯 **High accuracy**: Sub-percent precision across a wide parameter range
- 🔄 **Automatic differentiation**: Compute gradients with respect to cosmological parameters
- 🚀 **Batch processing**: Efficiently process multiple parameter sets simultaneously
- 🔧 **JAX integration**: Full compatibility with JAX transformations (JIT, vmap, grad)

## Features

### Speed
JaxCapse computes CMB power spectra orders of magnitude faster than traditional Boltzmann solvers:

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
JaxCapse provides trained emulators for:

- **TT**: Temperature-Temperature
- **EE**: E-mode polarization
- **TE**: Temperature-E-mode cross-correlation
- **PP**: Lensing potential

## Quick Example

```python
import jaxcapse
import jax.numpy as jnp

# Load a trained emulator
emulator_TT = jaxcapse.load_emulator("trained_emu/TT/")

# Define cosmological parameters
# [ln10As, ns, H0, ωb, ωc, τ]
params = jnp.array([3.1, 0.96, 67.0, 0.022, 0.12, 0.055])

# Compute power spectrum
cl_TT = emulator_TT.get_Cl(params)

# Compute gradients
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
- jaxace ≥ 0.1.1

## Citation

If you use JaxCapse in your research, please cite:

```bibtex
@software{jaxcapse2024,
  author = {Bonici, Marco},
  title = {JaxCapse: JAX implementation of CAPSE emulator},
  year = {2024},
  url = {https://github.com/CosmologicalEmulators/jaxcapse}
}
```

## License

JaxCapse is released under the MIT License. See [LICENSE](https://github.com/CosmologicalEmulators/jaxcapse/blob/main/LICENSE) for details.

## Acknowledgments

JaxCapse builds on:

- [jaxace](https://github.com/CosmologicalEmulators/jaxace): JAX implementation of AbstractCosmologicalEmulators.jl
- [CAPSE](https://github.com/CosmologicalEmulators/Capse.jl): Original Julia implementation
- [JAX](https://github.com/google/jax): Composable transformations of Python+NumPy programs