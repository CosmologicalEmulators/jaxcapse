# Batch Processing

Efficiently process multiple cosmologies simultaneously with JaxCapse.

## Basic Batch Processing

```python
import jaxcapse
import jax.numpy as jnp

# Load emulator
emulator = jaxcapse.load_emulator("trained_emu/TT/")

# Create batch of parameters
batch_params = jnp.array([
    [3.1, 0.96, 67.0, 0.022, 0.12, 0.055],
    [3.0, 0.97, 68.0, 0.023, 0.11, 0.060],
    [3.2, 0.95, 66.0, 0.021, 0.13, 0.050]
])

# Process batch
cl_batch = emulator.get_Cl_batch(batch_params)
print(f"Output shape: {cl_batch.shape}")  # (3, n_ell)
```

## Performance Benefits

```python
import time

# Create large batch
n_cosmologies = 1000
batch = jnp.tile([3.1, 0.96, 67.0, 0.022, 0.12, 0.055], (n_cosmologies, 1))

# Time batch processing
start = time.perf_counter()
cl_batch = emulator.get_Cl_batch(batch)
batch_time = time.perf_counter() - start

# Time individual processing
start = time.perf_counter()
cl_individual = jnp.array([emulator.get_Cl(p) for p in batch])
individual_time = time.perf_counter() - start

print(f"Batch: {batch_time:.3f}s")
print(f"Individual: {individual_time:.3f}s")
print(f"Speedup: {individual_time/batch_time:.1f}x")
```

## Memory Management

```python
def process_large_dataset(params, emulator, chunk_size=1000):
    """Process large dataset in chunks."""
    n_total = len(params)
    results = []
    
    for i in range(0, n_total, chunk_size):
        chunk = params[i:i+chunk_size]
        cl_chunk = emulator.get_Cl_batch(chunk)
        results.append(cl_chunk)
    
    return jnp.concatenate(results, axis=0)
```

## Next Steps

- [JAX Features](jax_features.md): Advanced JAX functionality