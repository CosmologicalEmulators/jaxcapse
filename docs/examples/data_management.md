# Data Management

The `jaxcapse` package includes a simple data fetcher for downloading and caching trained emulator files from Zenodo.

## Basic Usage

### Loading an Emulator

The simplest way to load an emulator is using the pre-loaded emulators:

```python
import jaxcapse

# Emulators are automatically downloaded and loaded on import
# Access them from the trained_emulators dictionary
emulators = jaxcapse.trained_emulators["camb_mnuw0wacdm"]
emulator_TT = emulators["TT"]
emulator_EE = emulators["EE"]
emulator_TE = emulators["TE"]
emulator_BB = emulators["BB"]
emulator_PP = emulators["PP"]

# Or load from a specific path
from jaxcapse import get_emulator_path, load_emulator
path = get_emulator_path("TT")  # Returns path to TT emulator
emulator = load_emulator(str(path))
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
fetcher = EmulatorDataFetcher(
    zenodo_url="https://zenodo.org/records/22921165/files/camb_mnuw0wacdm_500000_width96_runtime_v1.tar.xz?download=1",
    emulator_types=["TT", "TE", "EE", "BB", "PP"],
    cache_dir="/path/to/my/cache",
    expected_checksum="8f4ae21a0214bdf83ee5557b6d8369ed3db729b91f933e4550d6e2c6eb0b5af8"
)

# Load a specific emulator
emulator_path = fetcher.get_emulator_path("TT")
```

### Adding New Emulator Configurations

When you upload new emulator files to Zenodo, you can add them to jaxcapse:

```python
from jaxcapse import add_emulator_config

# Add a new emulator configuration
add_emulator_config(
    model_name="my_custom_model",
    zenodo_url="https://zenodo.org/record/1234567/files/my_emulators.tar.gz",
    emulator_types=["TT", "EE"],
    description="My custom LCDM emulators",
    checksum="abc123...",  # Optional SHA256 checksum
    auto_load=True  # Automatically download and load
)

# Access the newly loaded emulators
emulator_TT = jaxcapse.trained_emulators["my_custom_model"]["TT"]
```

### Manual Download Control

```python
from jaxcapse.data_fetcher import get_fetcher

fetcher = get_fetcher()

# Get path without downloading
path = fetcher.get_emulator_path("TT", download_if_missing=False)
if path is None:
    print("Emulator not cached")

# Force download even if cached
success = fetcher.download_and_extract(force=True, show_progress=True)
```

### Cache Management

```python
# Clear specific emulator from cache
fetcher.clear_cache("TT")  # Clear TT emulator

# Clear all cached files
fetcher.clear_cache()  # Clear everything
```

## Example: Computing Cl Jacobians

Here's a complete example that downloads emulator data and computes Jacobians:

```python
import jax
import jax.numpy as jnp
import jaxcapse

# Access the pre-loaded TT emulator
emulator_TT = jaxcapse.trained_emulators["camb_mnuw0wacdm"]["TT"]

# Define cosmological parameters
# Order: ln10As, ns, tau, H0, omega_b, omega_c, Mnu, w0, wa
fiducial_params = jnp.array([3.044, 0.965, 0.054, 67.4, 0.02237, 0.12, 0.06, -1.0, 0.0])

# The emulator has a predict method that works with JAX
cl_tt = emulator_TT.predict(fiducial_params)

# Compute Jacobian using JAX autodiff
jacobian_fn = jax.jacobian(emulator_TT.predict)
jacobian = jacobian_fn(fiducial_params)

# jacobian shape: (n_ell, n_params)
print(f"Cl shape: {cl_tt.shape}")
print(f"Jacobian shape: {jacobian.shape}")

# Example: sensitivity at ell=100
ell_index = 100
param_names = ['ln10As', 'ns', 'tau', 'H0', 'omega_b', 'omega_c', 'Mnu', 'w0', 'wa']
print(f"\nParameter sensitivities at ell={ell_index}:")
for i, param in enumerate(param_names):
    print(f"  ∂Cl/∂{param:8s} = {jacobian[ell_index, i]:+.3e}")
```

## Configuration for CI/CD

For GitHub Actions or other CI systems, you can set up automatic data fetching:

```yaml
# .github/workflows/docs.yml
- name: Cache emulator data
  uses: actions/cache@v3
  with:
    path: ~/.jaxcapse_data
    key: emulator-data-${{ hashFiles('**/pyproject.toml') }}
    restore-keys: |
      emulator-data-

- name: Download emulator data
  run: |
    python -c "
    from jaxcapse.data_fetcher import get_fetcher
    fetcher = get_fetcher()
    # Pre-download all emulators
    fetcher.download_and_extract(show_progress=True)
    "
```

## Setting Up Your Own Zenodo Dataset

1. **Upload your emulator files to Zenodo:**
   - Go to [Zenodo](https://zenodo.org)
   - Create a new upload
   - Package your emulator files as a tar.gz archive
   - Publish to get a DOI and download URL

2. **Add your emulators to jaxcapse:**
   ```python
   from jaxcapse import add_emulator_config

   # Use the direct tar.gz download URL from Zenodo
   add_emulator_config(
       model_name="my_model",
       zenodo_url="https://zenodo.org/record/XXXXX/files/my_emulators.tar.gz",
       emulator_types=["TT", "EE", "TE", "PP"],
       description="My custom emulators"
   )
   ```

3. **Optional: Add checksum for verification:**
   ```bash
   # Generate SHA256 checksum
   sha256sum my_emulators.tar.gz
   ```

   Then include it:
   ```python
   add_emulator_config(
       model_name="my_model",
       zenodo_url="https://zenodo.org/record/XXXXX/files/my_emulators.tar.gz",
       emulator_types=["TT", "EE", "TE", "PP"],
       checksum="abc123def456...",
       auto_load=True
   )
   ```
