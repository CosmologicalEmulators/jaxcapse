# jaxcapse
[![codecov](https://codecov.io/gh/CosmologicalEmulators/jaxcapse/graph/badge.svg?token=D6VJ14G46U)](https://codecov.io/gh/CosmologicalEmulators/jaxcapse)
[![arXiv](https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg)](https://arxiv.org/abs/2307.14339)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/jaxcapse/stable/)
[![Documentation Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://cosmologicalemulators.github.io/jaxcapse/dev/)

Repo containing the jaxcapse emulator.

## Documentation

- **[Stable Documentation](https://cosmologicalemulators.github.io/jaxcapse/stable/)** - Latest release documentation
- **[Development Documentation](https://cosmologicalemulators.github.io/jaxcapse/dev/)** - Latest development version documentation

## Installation and usage

In order to install `jaxcapse`, you can just run

```bash
pip install jaxcapse
```

If you prefer to use the latest version from the repository, you can clone it, enter it, and run

```bash
pip install .
```

In order to use the emulators, you have to import `jaxcapse` and load a trained emulator

```python3
import jaxcapse
import jax.numpy as np
trained_emu = jaxcapse.load_emulator("/path/to/emu/")
```
Then you are good to! You have to create an input array and retrieve your calculation result

```python3
input_array = np.array([...]) #write in the relevant numbers
result = trained_emu.get_Cl(input_array)

# The returned grid always matches the prediction. Subsampled emulators with
# at most 2048 knots are transparently interpolated with a cubic spline.
ell = trained_emu.get_ell_grid()
ell_training = trained_emu.get_training_ell_grid()
```

For automatically interpolated grids, source bounds within `0.1` of an integer
are snapped to that integer. Bounds farther away are moved inward.

## CAMB Mnu-w0-wa-CDM emulators

The published [CAMB + CosmoRec artifact](https://doi.org/10.5281/zenodo.22921165)
contains TT, TE, EE, BB, and PP models. jaxcapse downloads them into a separate
model-specific cache so they cannot be confused with the existing LCDM models:

```python
import jax.numpy as jnp
import jaxcapse

params = jnp.array([3.044, 0.965, 0.054, 67.4, 0.02237, 0.120, 0.06, -1.0, 0.0])
tt = jaxcapse.trained_emulators["camb_mnuw0wacdm"]["TT"]
D_ell_TT = tt.get_Cl(params)
ell = tt.get_ell_grid()  # 2:9500
```

The input order is `ln10As, ns, tau, H0, omega_b, omega_c, Mnu, w0, wa`,
restricted to `w0 + wa < -0.5`. Exact `Mnu = 0` was not in the training set.
Despite the method name, TT/TE/EE/BB outputs are lensed **D_ell in microK^2**;
PP is `[ell*(ell+1)]^2 C_ell^phiphi/(2*pi)` (dimensionless).

For a more detailed explanation, check the tutorial in the `notebooks` folder, which also shows a comparison with the standard `CAMB` Boltzmann solver.

## Citing

Free usage of the software in this repository is provided, given that you cite our release paper.

M. Bonici, F. Bianchini, J. Ruiz-Zapatero, [_Capse: efficient and auto-differentiable CMB power spectra emulation_](https://arxiv.org/abs/2307.14339)
