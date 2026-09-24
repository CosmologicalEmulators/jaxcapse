"""
Data fetcher for jaxcapse emulator files from Zenodo.

This module handles downloading, extracting, and caching of trained emulator data.
"""

import hashlib
import os
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional, Union
from urllib.error import URLError


class EmulatorDataFetcher:
    """
    Manages downloading and caching of emulator data from Zenodo.

    The data is cached in ~/.jaxcapse_data/ by default.
    """

    def __init__(self,
                 zenodo_url: str,
                 emulator_types: list,
                 cache_dir: Optional[Union[str, Path]] = None,
                 expected_checksum: Optional[str] = None):
        """
        Initialize the data fetcher.

        Parameters
        ----------
        zenodo_url : str
            URL to download the emulator tar.gz file from.
        emulator_types : list
            List of emulator types to expect (e.g., ["TT", "EE", "TE", "PP"]).
        cache_dir : str or Path, optional
            Directory to cache downloaded files.
            Defaults to ~/.jaxcapse_data/
        expected_checksum : str, optional
            Expected SHA256 checksum of the downloaded file for verification.
        """
        # Store required parameters
        self.zenodo_url = zenodo_url
        self.emulator_types = emulator_types
        self.expected_checksum = expected_checksum

        if cache_dir is None:
            self.cache_dir = Path.home() / ".jaxcapse_data"
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Path for the downloaded tar.gz file
        # Extract filename from URL
        tar_filename = self.zenodo_url.split('/')[-1].split('?')[0]
        self.tar_path = self.cache_dir / tar_filename

        # Path for extracted emulators
        self.emulators_dir = self.cache_dir / "emulators"

    def _download_file(self, url: str, destination: Path,
                      show_progress: bool = True) -> bool:
        """
        Download a file from URL to destination.

        Parameters
        ----------
        url : str
            URL to download from
        destination : Path
            Local path to save the file
        show_progress : bool
            Whether to show download progress

        Returns
        -------
        bool
            True if download successful, False otherwise
        """
        try:
            # Create temporary file for download
            temp_file = destination.with_suffix('.tmp')

            def download_hook(block_num, block_size, total_size):
                if show_progress and total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    mb_downloaded = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                          end='', flush=True)

            if show_progress:
                print(f"Downloading emulator data from Zenodo...")

            urllib.request.urlretrieve(url, temp_file,
                                      reporthook=download_hook if show_progress else None)

            if show_progress:
                print()  # New line after progress

            # Move temp file to final destination
            shutil.move(str(temp_file), str(destination))
            return True

        except (URLError, IOError) as e:
            if show_progress:
                print(f"\nError downloading: {e}")
            # Clean up temp file if exists
            temp_file = destination.with_suffix('.tmp')
            if temp_file.exists():
                temp_file.unlink()
            return False

    def _extract_tar(self, tar_path: Path, extract_to: Path,
                    show_progress: bool = True) -> bool:
        """
        Extract a gzip- or xz-compressed emulator archive.

        Parameters
        ----------
        tar_path : Path
            Path to the tar.gz file
        extract_to : Path
            Directory to extract files to
        show_progress : bool
            Whether to show extraction progress

        Returns
        -------
        bool
            True if extraction successful, False otherwise
        """
        try:
            if show_progress:
                print(f"Extracting emulator data...")

            extract_to.mkdir(parents=True, exist_ok=True)

            with tarfile.open(tar_path, 'r:*') as tar:
                root = extract_to.resolve()
                for member in tar.getmembers():
                    target = (root / member.name).resolve()
                    if not (member.isfile() or member.isdir()) or not target.is_relative_to(root):
                        raise tarfile.TarError(f"Unsafe archive member: {member.name}")
                extract_options = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
                tar.extractall(extract_to, **extract_options)

            if show_progress:
                print("Extraction complete!")

            return True

        except (tarfile.TarError, IOError) as e:
            if show_progress:
                print(f"Error extracting tar file: {e}")
            return False

    def _verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """
        Verify SHA256 checksum of a file.

        Parameters
        ----------
        filepath : Path
            Path to the file to verify
        expected_checksum : str
            Expected SHA256 checksum

        Returns
        -------
        bool
            True if checksum matches, False otherwise
        """
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest() == expected_checksum

    def download_and_extract(self, force: bool = False,
                           show_progress: bool = True) -> bool:
        """
        Download and extract emulator data if not already present.

        Parameters
        ----------
        force : bool
            Force re-download even if data exists
        show_progress : bool
            Whether to show progress

        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Check if emulators are already extracted
        if not force and self.emulators_dir.exists():
            # Check if all expected emulator directories exist
            all_exist = all(
                (self.emulators_dir / emulator_type).exists()
                for emulator_type in self.emulator_types
            )
            if all_exist:
                if show_progress:
                    print("Emulator data already available.")
                return True

        # Download tar file if needed
        if force or not self.tar_path.exists():
            if show_progress:
                print(f"Downloading from Zenodo...")
            success = self._download_file(self.zenodo_url, self.tar_path,
                                         show_progress=show_progress)
            if not success:
                return False

        # Verify downloaded and pre-existing archives before extraction.
        if self.expected_checksum:
            if show_progress:
                print("Verifying checksum...")
            if not self._verify_checksum(self.tar_path, self.expected_checksum):
                if show_progress:
                    print("ERROR: Checksum verification failed!")
                    print("The cached archive may be corrupted.")
                if self.tar_path.exists():
                    self.tar_path.unlink()
                return False
            elif show_progress:
                print("✓ Checksum verified")

        # Extract tar file
        if show_progress:
            print("Extracting emulator data...")
        success = self._extract_tar(self.tar_path, self.emulators_dir,
                                   show_progress=show_progress)

        if success:
            # Check if there's a nested 'trained_emu' directory
            trained_emu_nested = self.emulators_dir / "trained_emu"
            if trained_emu_nested.exists():
                # Move contents up one level
                if show_progress:
                    print("Reorganizing extracted files...")
                for item in trained_emu_nested.iterdir():
                    target = self.emulators_dir / item.name
                    if target.exists():
                        shutil.rmtree(target) if target.is_dir() else target.unlink()
                    shutil.move(str(item), str(target))
                # Remove the now-empty nested directory
                trained_emu_nested.rmdir()

            # Verify extraction
            found_emulators = []
            for emulator_type in self.emulator_types:
                emulator_dir = self.emulators_dir / emulator_type
                if emulator_dir.exists():
                    found_emulators.append(emulator_type)
                elif show_progress:
                    print(f"Warning: {emulator_type} emulator directory not found after extraction")

            if found_emulators:
                if show_progress:
                    print(f"✓ Found emulators: {', '.join(found_emulators)}")
                return True
            else:
                if show_progress:
                    print("Error: No emulator directories found after extraction")
                    print("Checking extracted structure...")
                    # Debug: show what was actually extracted
                    if self.emulators_dir.exists():
                        items = list(self.emulators_dir.iterdir())
                        print(f"Found in {self.emulators_dir}:")
                        for item in items[:10]:
                            print(f"  - {item.name}")
                return False

        return success

    def get_emulator_path(self, emulator_type: str,
                         download_if_missing: bool = True) -> Optional[Path]:
        """
        Get the path to an emulator directory.

        Parameters
        ----------
        emulator_type : str
            Type of emulator (TT, EE, TE, or PP)
        download_if_missing : bool
            Whether to download the data if not cached

        Returns
        -------
        Path or None
            Path to the emulator directory, or None if not available
        """
        if emulator_type not in self.emulator_types:
            raise ValueError(f"Unknown emulator type: {emulator_type}. "
                           f"Available types: {self.emulator_types}")

        emulator_path = self.emulators_dir / emulator_type

        if emulator_path.exists():
            return emulator_path

        # Download and extract if requested
        if download_if_missing:
            success = self.download_and_extract()
            if success and emulator_path.exists():
                return emulator_path

        return None

    def load_emulator(self, emulator_type: str, **kwargs) -> Any:
        """
        Load an emulator, downloading if necessary.

        Parameters
        ----------
        emulator_type : str
            Type of emulator to load (TT, EE, TE, or PP)
        **kwargs
            Additional arguments passed to the emulator loader

        Returns
        -------
        emulator
            The loaded emulator object (directory path)
        """
        emulator_path = self.get_emulator_path(emulator_type)
        if emulator_path is None:
            raise RuntimeError(f"Could not load {emulator_type} emulator")

        # For jaxcapse, return the directory path itself
        # The actual loading happens via jaxcapse.load_emulator(path)
        return str(emulator_path)

    def list_available(self) -> Dict[str, str]:
        """
        List all available emulator types.

        Returns
        -------
        dict
            Dictionary of emulator types and their descriptions
        """
        return {
            "TT": "CMB temperature power spectrum",
            "EE": "CMB E-mode polarization power spectrum",
            "TE": "CMB temperature-polarization cross spectrum",
            "BB": "CMB B-mode polarization power spectrum",
            "PP": "CMB lensing potential power spectrum"
        }

    def list_cached(self) -> list:
        """
        List all cached emulator directories.

        Returns
        -------
        list
            List of cached emulator types
        """
        cached = []
        if self.emulators_dir.exists():
            for emulator_type in self.emulator_types:
                if (self.emulators_dir / emulator_type).exists():
                    cached.append(emulator_type)
        return cached

    def clear_cache(self, emulator_type: Optional[str] = None):
        """
        Clear cached emulator files.

        Parameters
        ----------
        emulator_type : str, optional
            Specific emulator to remove from cache.
            If None, clears all cached files.
        """
        if emulator_type:
            if emulator_type in self.emulator_types:
                emulator_path = self.emulators_dir / emulator_type
                if emulator_path.exists():
                    shutil.rmtree(emulator_path)
                    print(f"Removed {emulator_type} from cache")
        else:
            # Clear all cached files
            if self.emulators_dir.exists():
                shutil.rmtree(self.emulators_dir)
            if self.tar_path.exists():
                self.tar_path.unlink()
            print("Cleared all cached emulator files")


# Convenience functions for direct access
_default_fetcher = None


def get_fetcher(zenodo_url: str = None,
                emulator_types: list = None,
                cache_dir: Optional[Union[str, Path]] = None,
                expected_checksum: str = None,
                model_name: str = None) -> EmulatorDataFetcher:
    """
    Get the default fetcher, reusing its singleton only for a default call.
    Explicitly configured calls always receive an independent fetcher.

    Parameters
    ----------
    zenodo_url : str, optional
        URL to download the emulator tar.gz file from.
        If None, uses the default jaxcapse URL.
    emulator_types : list, optional
        List of emulator types to expect.
        If None, uses default ["TT", "TE", "EE", "BB", "PP"].
    cache_dir : str or Path, optional
        Root cache directory. If ``model_name`` is provided, the model name is
        appended as a subdirectory.
    expected_checksum : str, optional
        Expected SHA256 checksum of the downloaded file.
        If None, uses the default checksum for the default URL.
    model_name : str, optional
        Name of the model-specific cache directory. The default fetcher uses
        the bundled ``camb_mnuw0wacdm`` model.

    Returns
    -------
    EmulatorDataFetcher
        The fetcher instance
    """
    global _default_fetcher
    use_default_fetcher = all(value is None for value in (
        zenodo_url, emulator_types, cache_dir, expected_checksum, model_name,
    ))

    if use_default_fetcher:
        if _default_fetcher is None:
            zenodo_url = "https://zenodo.org/records/22921165/files/camb_mnuw0wacdm_500000_width96_runtime_v1.tar.xz?download=1"
            expected_checksum = "8f4ae21a0214bdf83ee5557b6d8369ed3db729b91f933e4550d6e2c6eb0b5af8"
            emulator_types = ["TT", "TE", "EE", "BB", "PP"]
            cache_dir = Path.home() / ".jaxcapse_data" / "camb_mnuw0wacdm"
            _default_fetcher = EmulatorDataFetcher(
                zenodo_url, emulator_types, cache_dir, expected_checksum,
            )
        return _default_fetcher

    if zenodo_url is None:
        if model_name not in (None, "camb_mnuw0wacdm"):
            raise ValueError("zenodo_url is required for a model other than camb_mnuw0wacdm")
        zenodo_url = "https://zenodo.org/records/22921165/files/camb_mnuw0wacdm_500000_width96_runtime_v1.tar.xz?download=1"
        if expected_checksum is None:
            expected_checksum = "8f4ae21a0214bdf83ee5557b6d8369ed3db729b91f933e4550d6e2c6eb0b5af8"
    if emulator_types is None:
        emulator_types = ["TT", "TE", "EE", "BB", "PP"]

    base = Path(cache_dir) if cache_dir is not None else Path.home() / ".jaxcapse_data"
    if model_name is not None:
        base = base / model_name
    return EmulatorDataFetcher(zenodo_url, emulator_types, base, expected_checksum)


def get_emulator_directory(emulator_type: str) -> str:
    """
    Convenience function to get the emulator directory path using the default fetcher.

    Parameters
    ----------
    emulator_type : str
        Type of emulator (TT, EE, TE, or PP)

    Returns
    -------
    str
        Path to the emulator directory
    """
    return get_fetcher().load_emulator(emulator_type)


def get_emulator_path(emulator_type: str) -> Optional[Path]:
    """
    Get the path to an emulator directory.

    Parameters
    ----------
    emulator_type : str
        Type of emulator (TT, EE, TE, or PP)

    Returns
    -------
    Path or None
        Path to the emulator directory
    """
    return get_fetcher().get_emulator_path(emulator_type)
