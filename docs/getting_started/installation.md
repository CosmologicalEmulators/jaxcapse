# Installation

## Requirements

JaxCapse requires Python 3.10 or later. The main dependencies are:

- JAX ≥ 0.4.30
- Flax ≥ 0.10.0
- jaxace ≥ 0.1.1
- NumPy

## Install from PyPI

The simplest way to install JaxCapse is via pip:

```bash
pip install jaxcapse
```

## Install from Source

For the latest development version, install directly from GitHub:

```bash
git clone https://github.com/CosmologicalEmulators/jaxcapse.git
cd jaxcapse
pip install -e .
```

## Install with Poetry

If you're using Poetry for dependency management:

```bash
git clone https://github.com/CosmologicalEmulators/jaxcapse.git
cd jaxcapse
poetry install
```

## GPU Support

JaxCapse automatically uses GPU acceleration if available. To install JAX with GPU support:

### CUDA 11.8
```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### CUDA 12
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Verify Installation

To verify that JaxCapse is correctly installed:

```python
import jaxcapse
import jax

print(f"JaxCapse version: {jaxcapse.__version__}")
print(f"JAX version: {jax.__version__}")
print(f"Device: {jax.devices()[0]}")
```

## Getting Trained Emulators

JaxCapse requires trained neural network weights to function. You can:

1. **Download pre-trained emulators** from the [JaxCapse repository](https://github.com/CosmologicalEmulators/jaxcapse/tree/main/trained_emu)

2. **Train your own emulators** using the training scripts (see [Development Guide](../development/contributing.md))

Example structure for trained emulators:
```
trained_emu/
├── TT/
│   ├── nn_setup.json
│   ├── weights.npy
│   ├── inminmax.npy
│   ├── outminmax.npy
│   └── postprocessing.py
├── EE/
│   └── ...
├── TE/
│   └── ...
└── PP/
    └── ...
```

## Troubleshooting

### ImportError: No module named 'jaxcapse'

Make sure you've installed JaxCapse in your current Python environment:
```bash
pip show jaxcapse
```

### JAX not using GPU

Verify JAX can see your GPU:
```python
import jax
print(jax.devices())  # Should show GPU device
```

If not, reinstall JAX with the appropriate CUDA version.

### Memory errors with large batches

JaxCapse is memory-efficient, but very large batches may exceed GPU memory. Try:

- Reducing batch size
- Using CPU for extremely large batches
- Processing in chunks

## Next Steps

Once installed, proceed to the [Quick Start](quickstart.md) guide to begin using JaxCapse.