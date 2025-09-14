"""
Test error scenarios for data fetcher using mocks.
These tests mock network and file errors without real downloads.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import URLError
import tarfile
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxcapse.data_fetcher import EmulatorDataFetcher


class TestErrorScenarios(unittest.TestCase):
    """Tests for error conditions using mocks."""

    def setUp(self):
        """Set up test directory."""
        self.test_cache = tempfile.mkdtemp(prefix="jaxcapse_error_test_")
        self.zenodo_url = "https://zenodo.org/records/17115001/files/trained_emu.tar.gz?download=1"
        self.emulator_types = ["TT", "TE", "EE", "PP"]

    def tearDown(self):
        """Clean up test directory."""
        if Path(self.test_cache).exists():
            shutil.rmtree(self.test_cache)

    @patch('urllib.request.urlretrieve')
    def test_network_error_handling(self, mock_urlretrieve):
        """Test handling of network errors during download."""
        # Mock network error
        mock_urlretrieve.side_effect = URLError("Network error")

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Should return False on network error
        success = fetcher.download_and_extract(show_progress=False)
        self.assertFalse(success)

        # No files should exist
        self.assertFalse(fetcher.tar_path.exists())
        self.assertFalse(fetcher.emulators_dir.exists())

    @patch('urllib.request.urlretrieve')
    def test_ioerror_during_download(self, mock_urlretrieve):
        """Test handling of IOError during download."""
        # Mock IO error
        mock_urlretrieve.side_effect = IOError("Disk full")

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Should return False on IO error
        success = fetcher.download_and_extract(show_progress=False)
        self.assertFalse(success)

        # Temp file should be cleaned up
        temp_file = fetcher.tar_path.with_suffix('.tmp')
        self.assertFalse(temp_file.exists())

    def test_corrupted_tar_handling(self):
        """Test handling of corrupted tar file."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Create a corrupted tar file
        fetcher.tar_path.parent.mkdir(parents=True, exist_ok=True)
        fetcher.tar_path.write_bytes(b"This is not a valid tar file")

        # Mock download to skip actual download
        with patch.object(fetcher, '_download_file', return_value=True):
            # Extraction should fail
            success = fetcher._extract_tar(
                fetcher.tar_path,
                fetcher.emulators_dir,
                show_progress=False
            )
            self.assertFalse(success)

    @patch('tarfile.open')
    def test_extraction_error_handling(self, mock_tarfile_open):
        """Test handling of errors during extraction."""
        # Mock extraction error
        mock_tar = MagicMock()
        mock_tar.__enter__ = MagicMock(return_value=mock_tar)
        mock_tar.__exit__ = MagicMock(return_value=None)
        mock_tar.extractall.side_effect = tarfile.TarError("Extraction failed")
        mock_tarfile_open.return_value = mock_tar

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Create dummy tar file
        fetcher.tar_path.parent.mkdir(parents=True, exist_ok=True)
        fetcher.tar_path.touch()

        # Extraction should fail
        success = fetcher._extract_tar(
            fetcher.tar_path,
            fetcher.emulators_dir,
            show_progress=False
        )
        self.assertFalse(success)

    @patch('urllib.request.urlretrieve')
    def test_download_hook_with_progress(self, mock_urlretrieve):
        """Test download progress hook functionality."""
        # Track hook calls
        hook_calls = []

        def capture_hook(*args, **kwargs):
            if 'reporthook' in kwargs and kwargs['reporthook']:
                # Simulate progress callbacks
                hook = kwargs['reporthook']
                hook(0, 1024, 5242880)  # 0%
                hook(2560, 1024, 5242880)  # 50%
                hook(5120, 1024, 5242880)  # 100%
                hook_calls.append(True)
            return None

        mock_urlretrieve.side_effect = capture_hook

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Download with progress
        fetcher._download_file(
            self.zenodo_url,
            fetcher.tar_path,
            show_progress=True
        )

        # Hook should have been called
        self.assertTrue(len(hook_calls) > 0)

    def test_checksum_verification_removes_bad_file(self):
        """Test that failed checksum removes the downloaded file."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache,
            expected_checksum="0000000000000000000000000000000000000000000000000000000000000000"
        )

        # Mock the download to create a dummy file
        def mock_download_func(url, path, show_progress=False):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"dummy content for testing")
            return True

        with patch.object(fetcher, '_download_file', side_effect=mock_download_func):
            # Should fail due to bad checksum and remove file
            success = fetcher.download_and_extract(show_progress=False)
            self.assertFalse(success)
            # File should be removed after checksum failure
            self.assertFalse(fetcher.tar_path.exists())

    def test_get_emulator_path_invalid_type(self):
        """Test get_emulator_path with invalid emulator type."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Should raise ValueError for unknown type
        with self.assertRaises(ValueError) as context:
            fetcher.get_emulator_path("INVALID_TYPE")

        self.assertIn("Unknown emulator type", str(context.exception))
        self.assertIn("INVALID_TYPE", str(context.exception))

    def test_load_emulator_download_failure(self):
        """Test load_emulator when download fails."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Mock download failure
        with patch.object(fetcher, 'download_and_extract', return_value=False):
            # Should raise RuntimeError
            with self.assertRaises(RuntimeError) as context:
                fetcher.load_emulator("TT")

            self.assertIn("Could not load TT emulator", str(context.exception))

    def test_clear_cache_nonexistent(self):
        """Test clearing cache when directories don't exist."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Clear cache should not fail even if nothing exists
        fetcher.clear_cache("TT")  # Specific
        fetcher.clear_cache()  # All

        # Should complete without errors
        self.assertFalse(fetcher.tar_path.exists())
        self.assertFalse(fetcher.emulators_dir.exists())


class TestURLParsing(unittest.TestCase):
    """Test URL parsing and edge cases."""

    def test_url_with_query_params(self):
        """Test URL parsing with query parameters."""
        test_cache = tempfile.mkdtemp(prefix="jaxcapse_url_test_")

        try:
            # URL with query parameters
            url = "https://example.com/path/to/file.tar.gz?download=1&token=abc123"
            fetcher = EmulatorDataFetcher(
                zenodo_url=url,
                emulator_types=["TT"],
                cache_dir=test_cache
            )

            # Should extract filename correctly
            self.assertEqual(fetcher.tar_path.name, "file.tar.gz")

        finally:
            if Path(test_cache).exists():
                shutil.rmtree(test_cache)

    def test_url_without_query_params(self):
        """Test URL parsing without query parameters."""
        test_cache = tempfile.mkdtemp(prefix="jaxcapse_url_test2_")

        try:
            # URL without query parameters
            url = "https://example.com/path/to/myfile.tar.gz"
            fetcher = EmulatorDataFetcher(
                zenodo_url=url,
                emulator_types=["TT"],
                cache_dir=test_cache
            )

            # Should extract filename correctly
            self.assertEqual(fetcher.tar_path.name, "myfile.tar.gz")

        finally:
            if Path(test_cache).exists():
                shutil.rmtree(test_cache)

    def test_cache_directory_creation(self):
        """Test that cache directory is created if it doesn't exist."""
        # Use a non-existent directory
        test_cache = Path(tempfile.gettempdir()) / "jaxcapse_test_create" / "nested" / "dir"

        try:
            fetcher = EmulatorDataFetcher(
                zenodo_url="https://example.com/test.tar.gz",
                emulator_types=["TT"],
                cache_dir=test_cache
            )

            # Directory should be created
            self.assertTrue(fetcher.cache_dir.exists())
            self.assertTrue(fetcher.cache_dir.is_dir())

        finally:
            # Clean up the nested directories
            if test_cache.exists():
                shutil.rmtree(test_cache.parent.parent)


if __name__ == "__main__":
    unittest.main()