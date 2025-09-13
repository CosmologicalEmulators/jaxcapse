# Loading Emulators

This guide covers how to load and configure JaxCapse emulators.

## Basic Loading

### Single Emulator

```python
import jaxcapse

# Load a single emulator
emulator_TT = jaxcapse.load_emulator("trained_emu/TT/")
```

### Multiple Emulators

```python
# Load all available emulators
emulators = {}
for spectrum in ['TT', 'EE', 'TE', 'PP']:
    path = f"trained_emu/{spectrum}/"
    emulators[spectrum] = jaxcapse.load_emulator(path)
```

## Directory Structure

Each emulator directory must contain:

```
trained_emu/TT/
├── nn_setup.json       # Network architecture
├── weights.npy         # Trained weights
├── inminmax.npy       # Input normalization
├── outminmax.npy      # Output normalization
└── postprocessing.py  # Postprocessing function
```

## Inspecting Emulator Properties

### Emulator Description

```python
# View metadata
desc = emulator_TT.emulator_description
print(f"Author: {desc['author']}")
print(f"Parameters: {desc['parameters']}")
print(f"Info: {desc['miscellanea']}")
```

### Parameter Bounds

```python
# Check training ranges
bounds = emulator_TT.in_MinMax
param_names = ['ln10As', 'ns', 'H0', 'ωb', 'ωc', 'τ']

for i, name in enumerate(param_names):
    min_val, max_val = bounds[i]
    print(f"{name}: [{min_val:.3f}, {max_val:.3f}]")
```

### Output Information

```python
# Check output normalization
out_bounds = emulator_TT.out_MinMax
print(f"Number of ℓ values: {len(out_bounds)}")
print(f"Output range: [{out_bounds.min():.2e}, {out_bounds.max():.2e}]")
```

## Error Handling

### Missing Files

```python
try:
    emulator = jaxcapse.load_emulator("path/to/emulator/")
except FileNotFoundError as e:
    print(f"Missing file: {e}")
    # Handle missing emulator files
```

### Invalid Configuration

```python
try:
    emulator = jaxcapse.load_emulator("corrupted/emulator/")
except json.JSONDecodeError:
    print("Invalid nn_setup.json file")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Custom Emulator Paths

### From Environment Variable

```python
import os

# Set custom path
emulator_dir = os.environ.get('JAXCAPSE_DATA', 'trained_emu')
emulator = jaxcapse.load_emulator(f"{emulator_dir}/TT/")
```

### From Configuration File

```python
import json

# Load paths from config
with open('config.json', 'r') as f:
    config = json.load(f)

emulator = jaxcapse.load_emulator(config['emulator_path'])
```

## Validating Emulators

### Quick Validation

```python
def validate_emulator(emulator):
    """Quick validation of loaded emulator."""
    import jax.numpy as jnp
    
    # Test with fiducial parameters
    test_params = jnp.array([3.05, 0.965, 67.36, 0.0224, 0.120, 0.054])
    
    try:
        # Compute spectrum
        cl = emulator.get_Cl(test_params)
        
        # Check output
        assert cl.shape[0] > 0, "Empty output"
        assert jnp.all(jnp.isfinite(cl)), "Non-finite values"
        assert jnp.all(cl >= 0), "Negative power spectrum"
        
        print("✓ Emulator validation passed")
        return True
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

# Validate
validate_emulator(emulator_TT)
```

### Comprehensive Testing

```python
def test_emulator_comprehensive(emulator):
    """Comprehensive emulator testing."""
    import jax
    import jax.numpy as jnp
    
    # Test parameters
    params = jnp.array([3.05, 0.965, 67.36, 0.0224, 0.120, 0.054])
    
    tests_passed = []
    
    # Test 1: Basic computation
    try:
        cl = emulator.get_Cl(params)
        tests_passed.append("Basic computation")
    except:
        pass
    
    # Test 2: Batch processing
    try:
        batch = jnp.tile(params, (10, 1))
        cl_batch = emulator.get_Cl_batch(batch)
        tests_passed.append("Batch processing")
    except:
        pass
    
    # Test 3: Gradients
    try:
        grad = jax.grad(lambda p: jnp.sum(emulator.get_Cl(p)))(params)
        tests_passed.append("Gradient computation")
    except:
        pass
    
    # Test 4: JIT compilation
    try:
        jit_fn = jax.jit(emulator.get_Cl)
        cl_jit = jit_fn(params)
        tests_passed.append("JIT compilation")
    except:
        pass
    
    print(f"Passed {len(tests_passed)}/4 tests:")
    for test in tests_passed:
        print(f"  ✓ {test}")
    
    return len(tests_passed) == 4
```

## Memory Management

### Loading Multiple Emulators

```python
import gc

# Load emulators efficiently
def load_all_emulators(base_path="trained_emu"):
    """Load all emulators with memory management."""
    emulators = {}
    
    for spectrum in ['TT', 'EE', 'TE', 'PP']:
        try:
            path = f"{base_path}/{spectrum}/"
            emulators[spectrum] = jaxcapse.load_emulator(path)
            print(f"Loaded {spectrum}")
        except Exception as e:
            print(f"Failed to load {spectrum}: {e}")
    
    # Force garbage collection
    gc.collect()
    
    return emulators
```

### Unloading Emulators

```python
# Clear emulator from memory
del emulator_TT
gc.collect()
```

## Advanced Loading Options

### Lazy Loading

```python
class LazyEmulatorLoader:
    """Load emulators on demand."""
    
    def __init__(self, base_path="trained_emu"):
        self.base_path = base_path
        self._emulators = {}
    
    def get(self, spectrum):
        """Get emulator, loading if necessary."""
        if spectrum not in self._emulators:
            path = f"{self.base_path}/{spectrum}/"
            self._emulators[spectrum] = jaxcapse.load_emulator(path)
        return self._emulators[spectrum]
    
    def unload(self, spectrum):
        """Unload specific emulator."""
        if spectrum in self._emulators:
            del self._emulators[spectrum]

# Usage
loader = LazyEmulatorLoader()
emulator_TT = loader.get('TT')  # Loads on first access
emulator_TT = loader.get('TT')  # Returns cached version
```

### Parallel Loading

```python
from concurrent.futures import ThreadPoolExecutor

def load_emulator_wrapper(args):
    """Wrapper for parallel loading."""
    spectrum, base_path = args
    try:
        path = f"{base_path}/{spectrum}/"
        return spectrum, jaxcapse.load_emulator(path)
    except:
        return spectrum, None

# Load in parallel
spectra = ['TT', 'EE', 'TE', 'PP']
base_path = "trained_emu"

with ThreadPoolExecutor(max_workers=4) as executor:
    args = [(s, base_path) for s in spectra]
    results = executor.map(load_emulator_wrapper, args)
    emulators = dict(results)

# Filter out failed loads
emulators = {k: v for k, v in emulators.items() if v is not None}
```

## Next Steps

- [Computing Spectra](computing.md): Use loaded emulators
- [Batch Processing](batch.md): Process multiple cosmologies
- [JAX Features](jax_features.md): Advanced functionality