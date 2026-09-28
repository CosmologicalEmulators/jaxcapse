"""Compare jaxcapse interpolation with saved Julia reference outputs."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxcapse import SplinePlan

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parents[1]
DATA = ROOT / "tests" / "data"


def main():
    inputs = np.loadtxt(DATA / "spline_reference_inputs.txt")
    outputs = np.loadtxt(DATA / "spline_reference_outputs.txt")
    training_grid = jnp.asarray(inputs[:, 0])
    values = jnp.asarray(inputs[:, 1:3])
    reference = outputs[::20, 3:5]

    result = np.asarray(SplinePlan(training_grid)(values))
    absolute = np.abs(result - reference)
    denominator = np.maximum(np.abs(reference), np.finfo(float).tiny)
    relative = absolute / denominator
    max_abs_index = tuple(map(int, np.unravel_index(absolute.argmax(), absolute.shape)))
    max_rel_index = tuple(map(int, np.unravel_index(relative.argmax(), relative.shape)))

    print("method       max abs      abs index     max rel      rel index  result")
    print(
        f"SplinePlan  {absolute.max():12.5e} {str(max_abs_index):>14s} "
        f"{relative.max():12.5e} {str(max_rel_index):>14s}  "
        f"{'PASS' if absolute.max() <= 1e-12 else 'FAIL'}"
    )
    if absolute.max() > 1e-12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
