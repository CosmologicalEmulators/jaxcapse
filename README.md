# jaxcapse
[![codecov](https://codecov.io/gh/CosmologicalEmulators/jaxcapse/graph/badge.svg?token=D6VJ14G46U)](https://codecov.io/gh/CosmologicalEmulators/jaxcapse)
[![arXiv](https://img.shields.io/badge/arXiv-2307.14339-b31b1b.svg)](https://arxiv.org/abs/2307.14339)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://cosmologicalemulators.github.io/jaxcapse/)

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
```

For a more detailed explanation, check the tutorial in the `notebooks` folder, which also shows a comparison with the standard `CAMB` Boltzmann solver.

## Citing

Free usage of the software in this repository is provided, given that you cite our release paper.

M. Bonici, F. Bianchini, J. Ruiz-Zapatero, [_Capse: efficient and auto-differentiable CMB power spectra emulation_](https://arxiv.org/abs/2307.14339)
