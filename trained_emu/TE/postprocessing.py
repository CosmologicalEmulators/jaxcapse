import jax.numpy as jnp

def postprocessing(input, Cl):
    return Cl * jnp.exp(input[0]-3)