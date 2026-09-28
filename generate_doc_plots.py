#!/usr/bin/env python
"""Generate documentation plots from the bundled CAMB Mnu-w0-wa emulators."""

import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
plt.rcParams["text.usetex"] = shutil.which("latex") is not None

PARAMS = [3.044, 0.965, 0.054, 67.4, 0.02237, 0.12, 0.06, -1.0, 0.0]
PARAMETER_NAMES = [
    r"\ln(10^{10}A_s)", r"n_s", r"\tau", r"H_0",
    r"\omega_b", r"\omega_c", r"M_\nu", r"w_0", r"w_a",
]
SPECTRA = ("TT", "TE", "EE", "BB", "PP")
OUTPUT_DIR = Path(os.environ.get("JAXCAPSE_DOC_PLOT_DIR", "docs/images"))


def _load_emulators():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    import jaxcapse

    emulators = jaxcapse.trained_emulators["camb_mnuw0wacdm"]
    if set(emulators) != set(SPECTRA) or any(emulators[name] is None for name in SPECTRA):
        raise RuntimeError("The published five-spectrum CAMB emulator is not fully loaded")
    params = jnp.asarray(PARAMS, dtype=jnp.float64)
    return jax, jnp, emulators, params


def _ell_grid(emulators):
    ell = np.asarray(emulators["TT"].get_ell_grid())
    if not np.array_equal(ell, np.arange(2, 9501)):
        raise ValueError("The bundled CAMB spectra must use ell=2..9500")
    return ell


def _save(fig, filename):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def generate_cmb_spectra_plot():
    """Plot the five lensed D_ell emulator outputs."""
    _, _, emulators, params = _load_emulators()
    ell = _ell_grid(emulators)
    spectra = {name: np.asarray(emulators[name].get_Cl(params)) for name in SPECTRA}

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    for ax, name in zip(axes.flat, SPECTRA):
        values = spectra[name]
        if name == "TE":
            ax.semilogx(ell, values)
        else:
            ax.loglog(ell, values)
        ax.set(title=name, xlabel=r"$\ell$", ylabel=r"$D_\ell$")
        ax.grid(alpha=0.3)
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    _save(fig, "cmb_spectra.png")


def generate_jacobian_plot():
    """Plot the TT Jacobian with respect to all nine cosmological inputs."""
    jax, _, emulators, params = _load_emulators()
    ell = _ell_grid(emulators)
    jacobian = np.asarray(jax.jacfwd(emulators["TT"].get_Cl)(params))

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for index, (ax, name) in enumerate(zip(axes.flat, PARAMETER_NAMES)):
        ax.semilogx(ell, jacobian[:, index])
        ax.set(xlabel=r"$\ell$", ylabel=rf"$\partial D_\ell^{{TT}}/\partial {name}$")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "jacobian_tt.png")


def generate_elasticities_plot():
    """Plot TT logarithmic sensitivities for all nine cosmological inputs."""
    jax, _, emulators, params = _load_emulators()
    ell = _ell_grid(emulators)
    emulator = emulators["TT"]
    values = np.asarray(emulator.get_Cl(params))
    jacobian = np.asarray(jax.jacfwd(emulator.get_Cl)(params))
    elasticities = np.divide(
        jacobian * np.asarray(params)[None, :],
        values[:, None],
        out=np.full_like(jacobian, np.nan),
        where=values[:, None] != 0,
    )

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for index, (ax, name) in enumerate(zip(axes.flat, PARAMETER_NAMES)):
        ax.semilogx(ell, elasticities[:, index])
        ax.set(
            xlabel=r"$\ell$",
            ylabel=rf"$({name}/D_\ell^{{TT}})\,\partial D_\ell^{{TT}}/\partial {name}$",
        )
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _save(fig, "elasticities_tt.png")


def main():
    generate_cmb_spectra_plot()
    generate_jacobian_plot()
    generate_elasticities_plot()


if __name__ == "__main__":
    main()
