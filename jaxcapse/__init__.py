"""
jaxcapse: JAX-based Cosmological Analysis for Power Spectrum Emulation

This package provides tools for emulating cosmological power spectra using JAX.
"""

from .data_fetcher import (
    get_emulator_path,
    get_fetcher,
    EmulatorDataFetcher
)
from .jaxcapse import load_emulator  # Use the actual load_emulator function from jaxcapse.py

__all__ = [
    "load_emulator",
    "get_emulator_path",
    "get_fetcher",
    "EmulatorDataFetcher",
    "trained_emulators",
    "EMULATOR_CONFIGS",
    "add_emulator_config",
    "reload_emulators",
]

__version__ = "0.1.2"

# Automatically download and extract emulators when package is imported
# This ensures data is available when users import jaxcapse
# Set JAXCAPSE_NO_AUTO_DOWNLOAD=1 to disable automatic downloading
import os
import warnings

# Initialize the trained_emulators dictionary
trained_emulators = {}

# Define available emulator configurations
# This can be easily extended with new models in the future
EMULATOR_CONFIGS = {
    "camb_lcdm": {
        "zenodo_url": "https://zenodo.org/records/17115001/files/trained_emu.tar.gz?download=1",
        "emulator_types": ["TT", "TE", "EE", "PP"],
        "description": "CAMB for the LCDM model"
    }
    # Future models can be added here:
    # "class_lcdm": {
    #     "zenodo_url": "https://zenodo.org/...",
    #     "emulator_types": ["TT", "EE"],
    #     "description": "Standard LCDM model"
    # }
}


def _load_emulator_set(model_name: str, config: dict, auto_download: bool = True):
    """
    Helper function to load a set of emulators for a given model.

    Parameters
    ----------
    model_name : str
        Name of the model (e.g., "class_mnuw0wacdm")
    config : dict
        Configuration dictionary with zenodo_url and emulator_types
    auto_download : bool
        Whether to automatically download if not cached

    Returns
    -------
    dict
        Dictionary of loaded emulators
    """
    emulators = {}

    try:
        # Initialize fetcher for this model
        fetcher = get_fetcher(
            zenodo_url=config["zenodo_url"],
            emulator_types=config["emulator_types"]
        )

        # Download if needed and requested
        if auto_download:
            cached = fetcher.list_cached()
            if not cached or len(cached) < len(config["emulator_types"]):
                print(f"jaxcapse: Downloading {model_name} emulators from Zenodo...")
                fetcher.download_and_extract(show_progress=True)

        # Load each emulator
        for emulator_type in config["emulator_types"]:
            try:
                emulator_path = get_emulator_path(emulator_type)
                if emulator_path and emulator_path.exists():
                    emulators[emulator_type] = load_emulator(str(emulator_path))
                else:
                    emulators[emulator_type] = None
                    warnings.warn(f"Could not find {emulator_type} emulator for {model_name}")
            except Exception as e:
                emulators[emulator_type] = None
                warnings.warn(f"Error loading {emulator_type} for {model_name}: {e}")

    except Exception as e:
        warnings.warn(f"Could not initialize {model_name}: {e}")
        # Create empty entries
        emulators = {emu_type: None for emu_type in config.get("emulator_types", [])}

    return emulators


# Load default emulators on import (unless disabled)
if not os.environ.get("JAXCAPSE_NO_AUTO_DOWNLOAD"):
    print("jaxcapse: Loading emulators into memory...")

    # Load all configured models
    for model_name, config in EMULATOR_CONFIGS.items():
        trained_emulators[model_name] = _load_emulator_set(
            model_name,
            config,
            auto_download=True
        )

        # Report loading status
        loaded = sum(1 for v in trained_emulators[model_name].values() if v is not None)
        total = len(config["emulator_types"])
        if loaded > 0:
            print(f"  {model_name}: Loaded {loaded}/{total} emulators")
else:
    # Create empty structure when auto-download is disabled
    for model_name, config in EMULATOR_CONFIGS.items():
        trained_emulators[model_name] = {
            emu_type: None for emu_type in config["emulator_types"]
        }


def add_emulator_config(model_name: str,
                        zenodo_url: str,
                        emulator_types: list,
                        description: str = None,
                        auto_load: bool = True):
    """
    Add a new emulator configuration and optionally load it.

    Parameters
    ----------
    model_name : str
        Name for the model (e.g., "class_lcdm")
    zenodo_url : str
        URL to download the emulator tar.gz file from
    emulator_types : list
        List of emulator types (e.g., ["TT", "EE"])
    description : str, optional
        Description of the model
    auto_load : bool, optional
        Whether to immediately load the emulators

    Returns
    -------
    dict
        The loaded emulators for this model
    """
    global EMULATOR_CONFIGS, trained_emulators

    # Add to configuration
    EMULATOR_CONFIGS[model_name] = {
        "zenodo_url": zenodo_url,
        "emulator_types": emulator_types,
        "description": description or f"{model_name} emulators"
    }

    # Load if requested
    if auto_load:
        print(f"Loading {model_name} emulators...")
        trained_emulators[model_name] = _load_emulator_set(
            model_name,
            EMULATOR_CONFIGS[model_name],
            auto_download=True
        )

        # Report status
        loaded = sum(1 for v in trained_emulators[model_name].values() if v is not None)
        total = len(emulator_types)
        print(f"  Loaded {loaded}/{total} emulators")
    else:
        # Create empty structure
        trained_emulators[model_name] = {
            emu_type: None for emu_type in emulator_types
        }

    return trained_emulators[model_name]


def reload_emulators(model_name: str = None):
    """
    Reload emulators for a specific model or all models.

    Parameters
    ----------
    model_name : str, optional
        Specific model to reload. If None, reloads all.

    Returns
    -------
    dict
        The trained_emulators dictionary
    """
    global trained_emulators

    if model_name:
        # Reload specific model
        if model_name in EMULATOR_CONFIGS:
            print(f"Reloading {model_name}...")
            trained_emulators[model_name] = _load_emulator_set(
                model_name,
                EMULATOR_CONFIGS[model_name],
                auto_download=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(EMULATOR_CONFIGS.keys())}")
    else:
        # Reload all models
        print("Reloading all emulators...")
        for name, config in EMULATOR_CONFIGS.items():
            trained_emulators[name] = _load_emulator_set(name, config, auto_download=True)

    return trained_emulators
