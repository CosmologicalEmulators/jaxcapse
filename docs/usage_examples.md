# Usage Examples

The bundled model is the CAMB 2.0.4 + CosmoRec `Mnu-w0-wa-CDM` emulator. On
import, jaxcapse loads its five spectra under
`jaxcapse.trained_emulators["camb_mnuw0wacdm"]`.

## Evaluate all spectra

```python
import jax
import jax.numpy as jnp
import jaxcapse

emulators = jaxcapse.trained_emulators["camb_mnuw0wacdm"]

# Parameter order: ln10As, ns, tau, H0, omega_b, omega_c, Mnu, w0, wa.
params = jnp.array([3.044, 0.965, 0.054, 67.4, 0.02237, 0.12, 0.06, -1.0, 0.0])

# Training bounds: [2.5, 0.85, 0.02, 50, 0.02, 0.08, 0, -3, -3]
#                  [3.5, 1.05, 0.15, 90, 0.025, 0.16, 0.5, 0.5, 2]
# Plus the conditional prior w0 + wa < -0.5.
lower = jnp.array([2.5, 0.85, 0.02, 50, 0.02, 0.08, 0, -3, -3])
upper = jnp.array([3.5, 1.05, 0.15, 90, 0.025, 0.16, 0.5, 0.5, 2])
assert bool(jnp.all((lower <= params) & (params <= upper)))
assert float(params[7] + params[8]) < -0.5

ell = emulators["TT"].get_ell_grid()
assert bool(jnp.array_equal(ell, jnp.arange(2, 9501)))
spectra = {name: emulator.get_Cl(params) for name, emulator in emulators.items()}
```

All five predictions have 9,499 samples at `ell=2..9500`. `TT`, `TE`, `EE` and
`BB` are lensed `D_ell` in μK². `PP` is
`[ell(ell+1)]² C_ell^phiphi/(2π)` (dimensionless), despite the `get_Cl` name.
The released training sample contains no exact `Mnu=0` point; prefer an
interior positive mass when evaluating the model.

For custom artifacts, `l.npy` must have exactly one multipole per network
output. `load_emulator` raises `ValueError` on a length mismatch; it does not
guess a slice for legacy grids.

## Plot the spectra

```python
import matplotlib.pyplot as plt
import numpy as np

ell = np.asarray(ell)
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
for ax, name in zip(axes.flat, ("TT", "TE", "EE", "BB", "PP")):
    values = np.asarray(spectra[name])
    if name == "TE":  # TE changes sign.
        ax.semilogx(ell, values)
    else:
        ax.loglog(ell, values)
    ax.set(title=name, xlabel=r"$\ell$", ylabel=r"$D_\ell$")
    ax.grid(alpha=0.3)
axes.flat[-1].set_visible(False)
fig.tight_layout()
plt.show()
```

## Batch evaluation

`get_Cl_batch` and `jax.vmap` both accept a batch with shape
`(n_samples, 9)`:

```python
params_batch = jnp.stack((params, params.at[6].set(0.1)))
tt = emulators["TT"]
batch_spectra = tt.get_Cl_batch(params_batch)
vmap_spectra = jax.vmap(tt.get_Cl)(params_batch)
assert batch_spectra.shape == (2, 9499)
assert bool(jnp.allclose(batch_spectra, vmap_spectra, rtol=1e-12))
```

## Gradients and Hessians

```python
tt = emulators["TT"]
tt_jacobian = jax.jacfwd(tt.get_Cl)(params)

def scalar_prediction(x):
    return jnp.sum(tt.get_Cl(x)[100:120])

tt_hessian = jax.hessian(scalar_prediction)(params)
```
