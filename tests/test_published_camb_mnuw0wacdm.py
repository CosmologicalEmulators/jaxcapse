"""Published CAMB Mnu-w0-wa artifact, checked against Julia text fixtures."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxcapse import load_emulator
from jaxcapse.data_fetcher import EmulatorDataFetcher, get_fetcher


URL = "https://zenodo.org/records/22921165/files/camb_mnuw0wacdm_500000_width96_runtime_v1.tar.xz?download=1"
CHECKSUM = "8f4ae21a0214bdf83ee5557b6d8369ed3db729b91f933e4550d6e2c6eb0b5af8"
SPECTRA = ("TT", "TE", "EE", "BB", "PP")
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def published_models(tmp_path_factory):
    fetcher = EmulatorDataFetcher(
        URL, list(SPECTRA), cache_dir=tmp_path_factory.mktemp("capse-published"),
        expected_checksum=CHECKSUM,
    )
    assert fetcher.download_and_extract(show_progress=False)
    return {s: load_emulator(str(fetcher.get_emulator_path(s, download_if_missing=False)))
            for s in SPECTRA}


def test_model_caches_are_isolated(tmp_path):
    new = get_fetcher(URL, list(SPECTRA), cache_dir=tmp_path,
                      expected_checksum=CHECKSUM, model_name="camb_mnuw0wacdm")
    assert new.emulators_dir == tmp_path / "camb_mnuw0wacdm" / "emulators"
    assert new.get_emulator_path("TT", download_if_missing=False) is None


def test_only_new_model_is_bundled():
    import jaxcapse
    assert "camb_mnuw0wacdm" in jaxcapse.EMULATOR_CONFIGS
    assert "camb_lcdm" not in jaxcapse.EMULATOR_CONFIGS
    assert "camb_mnuw0wacdm" in jaxcapse.trained_emulators
    assert "camb_lcdm" not in jaxcapse.trained_emulators


def test_published_models_match_reference(published_models):
    inputs = np.loadtxt(DATA / "camb_mnuw0wacdm_inputs.txt")
    multipoles = np.array([2, 20, 200, 1000, 3000, 5000, 9500])
    assert set(published_models) == set(SPECTRA)
    for line in (DATA / "camb_mnuw0wacdm_reference.txt").read_text().splitlines():
        if line.startswith("#"):
            continue
        spectrum, sample, *reference = line.split()
        expected = np.array([float(v) for v in reference])
        emulator = published_models[spectrum]
        np.testing.assert_array_equal(np.asarray(emulator.get_ell_grid()), np.arange(2, 9501))
        prediction = np.asarray(emulator.get_Cl(jnp.asarray(inputs[int(sample) - 1])))
        assert prediction.shape == (9499,) and np.isfinite(prediction).all()
        assert np.max(np.abs(prediction[multipoles - 2] - expected)) / np.max(np.abs(expected)) < 1e-12


def test_published_models_are_differentiable(published_models):
    x = jnp.asarray(np.loadtxt(DATA / "camb_mnuw0wacdm_inputs.txt")[0])
    for emulator in published_models.values():
        scale = jnp.max(jnp.abs(emulator.get_Cl(x)))
        gradient = jax.grad(lambda params: jnp.sum(emulator.get_Cl(params)[100:301]) / scale)(x)
        assert bool(jnp.all(jnp.isfinite(gradient)))
        assert bool(jnp.max(jnp.abs(gradient)) > 0)


def test_published_model_batch_matches_vmap(published_models):
    inputs = jnp.asarray(np.loadtxt(DATA / "camb_mnuw0wacdm_inputs.txt")[:2])
    for emulator in published_models.values():
        batch = emulator.get_Cl_batch(inputs)
        vectorized = jax.vmap(emulator.get_Cl)(inputs)
        assert batch.shape == (2, 9499)
        np.testing.assert_allclose(np.asarray(batch), np.asarray(vectorized), rtol=1e-13)


def test_published_model_hessian_is_finite_and_symmetric(published_models):
    emulator = published_models["TT"]
    params = jnp.asarray(np.loadtxt(DATA / "camb_mnuw0wacdm_inputs.txt")[0])
    scalar_prediction = lambda x: jnp.sum(emulator.get_Cl(x)[100:120])
    hessian = jax.hessian(scalar_prediction)(params)
    assert hessian.shape == (9, 9)
    assert bool(jnp.all(jnp.isfinite(hessian)))
    np.testing.assert_allclose(np.asarray(hessian), np.asarray(hessian.T), rtol=1e-12, atol=1e-12)
