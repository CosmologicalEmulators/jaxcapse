# Data Management

The `jaxcapse` package includes a simple data fetcher for downloading and caching trained emulator files from Zenodo.

## Basic Usage

### Loading an Emulator

The simplest way to load an emulator is using the convenience function:

```python
import jaxcapse

# This will automatically download the emulator from Zenodo if not cached
emulator = jaxcapse.load_emulator("CMB_TT_emulator")
```

### Checking Available Emulators

```python
from jaxcapse.data_fetcher import get_fetcher

fetcher = get_fetcher()

# List all available emulators
available = fetcher.list_available()
for name, description in available.items():
    print(f"{name}: {description}")

# List cached emulators
cached = fetcher.list_cached()
print(f"Cached emulators: {cached}")
```

## Advanced Usage

### Custom Cache Directory

```python
from jaxcapse.data_fetcher import EmulatorDataFetcher

# Use a custom cache directory
fetcher = EmulatorDataFetcher(cache_dir="/path/to/my/cache")
emulator = fetcher.load_emulator("CMB_TT_emulator")
```

### Updating Zenodo URLs

When you upload new emulator files to Zenodo, you can update the URLs:

```python
from jaxcapse.data_fetcher import update_zenodo_url

# Update with your actual Zenodo URL
update_zenodo_url(
    "CMB_TT_emulator",
    "https://zenodo.org/record/1234567/files/CMB_TT_emulator.pkl",
    checksum="abc123..."  # Optional SHA256 checksum
)
```

### Manual Download Control

```python
fetcher = get_fetcher()

# Get path without downloading
path = fetcher.get_emulator_path("CMB_TT_emulator", download_if_missing=False)
if path is None:
    print("Emulator not cached")

# Download without verification
path = fetcher.get_emulator_path("CMB_TT_emulator", verify_checksum=False)
```

### Cache Management

```python
# Clear specific emulator from cache
fetcher.clear_cache("CMB_TT_emulator")

# Clear all cached files
fetcher.clear_cache()
```

## Example: Computing Cl Jacobians

Here's a complete example that downloads emulator data and computes Jacobians:

```python
import jax
import jax.numpy as jnp
import jaxcapse
import matplotlib.pyplot as plt

# Load the CMB TT emulator (downloads from Zenodo if needed)
emulator = jaxcapse.load_emulator("CMB_TT_emulator")

# Define cosmological parameters
fiducial_params = {
    'omega_b': 0.02237,
    'omega_c': 0.1200,
    'h': 0.6736,
    'ln10As': 3.044,
    'ns': 0.9649,
    'tau': 0.0544
}

# Function to compute Cl from parameters
def cl_function(params_array):
    """Compute CMB power spectrum from parameter array."""
    omega_b, omega_c, h, ln10As, ns, tau = params_array

    # Create cosmology object
    cosmo = jaxcapse.Cosmology(
        omega_b=omega_b,
        omega_c=omega_c,
        h=h,
        ln10As=ln10As,
        ns=ns,
        tau=tau
    )

    # Run emulator to get Cl
    ell, cl_tt = emulator.predict(cosmo)
    return cl_tt

# Convert dict to array for JAX
params_array = jnp.array(list(fiducial_params.values()))

# Compute Jacobian
jacobian_fn = jax.jacobian(cl_function)
jacobian = jacobian_fn(params_array)

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
param_names = list(fiducial_params.keys())

for i, (ax, param) in enumerate(zip(axes.flat, param_names)):
    ax.plot(ell, jacobian[:, i])
    ax.set_xlabel(r'$\ell$')
    ax.set_ylabel(rf'$\partial C_\ell^{{TT}} / \partial {param}$')
    ax.set_title(f'Jacobian w.r.t. {param}')
    ax.set_xscale('log')

plt.tight_layout()
plt.savefig('cl_jacobian.png', dpi=150, bbox_inches='tight')
plt.show()
```

## Configuration for CI/CD

For GitHub Actions or other CI systems, you can set up automatic data fetching:

```yaml
# .github/workflows/docs.yml
- name: Cache emulator data
  uses: actions/cache@v3
  with:
    path: ~/.jaxcapse_data
    key: emulator-data-${{ hashFiles('**/registry.json') }}
    restore-keys: |
      emulator-data-

- name: Download emulator data
  run: |
    python -c "
    from jaxcapse.data_fetcher import get_fetcher
    fetcher = get_fetcher()
    # Pre-download all emulators needed for docs
    for emulator in ['CMB_TT_emulator', 'CMB_EE_emulator']:
        fetcher.get_emulator_path(emulator)
    "
```

## Setting Up Your Own Zenodo Dataset

1. **Upload your emulator files to Zenodo:**
   - Go to [Zenodo](https://zenodo.org)
   - Create a new upload
   - Add your `.pkl` files
   - Publish to get a DOI

2. **Update the registry in your code:**
   ```python
   from jaxcapse.data_fetcher import update_zenodo_url

   # Use the direct file download URL from Zenodo
   update_zenodo_url(
       "my_emulator",
       "https://zenodo.org/record/XXXXX/files/my_emulator.pkl"
   )
   ```

3. **Optional: Add checksum for verification:**
   ```bash
   # Generate SHA256 checksum
   sha256sum my_emulator.pkl
   ```

   Then include it:
   ```python
   update_zenodo_url(
       "my_emulator",
       "https://zenodo.org/record/XXXXX/files/my_emulator.pkl",
       checksum="abc123def456..."
   )
   ```