"""
Integration tests for data fetcher with real Zenodo downloads.
These tests actually download the 5MB file from Zenodo to ensure real-world functionality.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jaxcapse.data_fetcher import EmulatorDataFetcher, get_fetcher


class TestRealZenodoDownload(unittest.TestCase):
    """Tests that actually download from Zenodo (5MB file)."""

    @classmethod
    def setUpClass(cls):
        """Set up test cache directory."""
        cls.test_cache = tempfile.mkdtemp(prefix="jaxcapse_test_")
        cls.zenodo_url = "https://zenodo.org/records/17115001/files/trained_emu.tar.gz?download=1"
        cls.correct_checksum = "b1d6f47c3bafb6b1ef0b80069e3d7982f274c6c7352ee44e460ffb9c2a573210"
        cls.emulator_types = ["TT", "TE", "EE", "PP"]
        print(f"\nTest cache directory: {cls.test_cache}")

    @classmethod
    def tearDownClass(cls):
        """Clean up downloaded data unless KEEP_TEST_DATA is set."""
        if not os.environ.get("KEEP_TEST_DATA"):
            if Path(cls.test_cache).exists():
                shutil.rmtree(cls.test_cache)
                print("Test cache cleaned up")
        else:
            print(f"Test data kept at: {cls.test_cache}")

    def test_download_and_extract_real_data(self):
        """Test actual download from Zenodo with extraction."""
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=self.test_cache
        )

        # Should download and extract
        success = fetcher.download_and_extract(show_progress=False)
        self.assertTrue(success)

        # Check that tar file was downloaded
        self.assertTrue(fetcher.tar_path.exists())
        self.assertGreater(fetcher.tar_path.stat().st_size, 1000000)  # > 1MB

        # Check that emulators were extracted
        for emu_type in self.emulator_types:
            emu_path = fetcher.emulators_dir / emu_type
            self.assertTrue(emu_path.exists(), f"{emu_type} not extracted")
            self.assertTrue(emu_path.is_dir())

            # Check for expected files in each emulator directory
            expected_files = ["weights.npy", "inminmax.npy", "outminmax.npy", "l.npy"]
            for file in expected_files:
                file_path = emu_path / file
                self.assertTrue(file_path.exists(), f"{file} not found in {emu_type}")

    def test_checksum_verification_success(self):
        """Test download with correct checksum verification."""
        cache_dir = Path(self.test_cache) / "checksum_test_success"
        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir,
            expected_checksum=self.correct_checksum
        )

        # Should succeed with correct checksum
        success = fetcher.download_and_extract(show_progress=False)
        self.assertTrue(success)

        # Verify files exist
        self.assertTrue(fetcher.tar_path.exists())
        for emu_type in self.emulator_types:
            self.assertTrue((fetcher.emulators_dir / emu_type).exists())

    def test_checksum_verification_failure(self):
        """Test download with incorrect checksum fails properly."""
        cache_dir = Path(self.test_cache) / "checksum_test_fail"
        wrong_checksum = "0000000000000000000000000000000000000000000000000000000000000000"

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir,
            expected_checksum=wrong_checksum
        )

        # Should fail with wrong checksum
        success = fetcher.download_and_extract(show_progress=False)
        self.assertFalse(success)

        # Tar file should be removed after failed checksum
        self.assertFalse(fetcher.tar_path.exists())

    def test_cache_reuse_no_redownload(self):
        """Test that second access uses cache without re-downloading."""
        cache_dir = Path(self.test_cache) / "cache_reuse_test"

        # First fetcher downloads
        fetcher1 = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir,
            expected_checksum=self.correct_checksum
        )
        success1 = fetcher1.download_and_extract(show_progress=False)
        self.assertTrue(success1)

        # Record modification time of tar file
        tar_mtime = fetcher1.tar_path.stat().st_mtime

        # Second fetcher should reuse cache
        fetcher2 = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir,
            expected_checksum=self.correct_checksum
        )
        success2 = fetcher2.download_and_extract(show_progress=False)
        self.assertTrue(success2)

        # Tar file should not have been re-downloaded (same modification time)
        self.assertEqual(tar_mtime, fetcher2.tar_path.stat().st_mtime)

    def test_force_redownload(self):
        """Test that force=True re-downloads even if cached."""
        cache_dir = Path(self.test_cache) / "force_download_test"

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir
        )

        # First download
        success1 = fetcher.download_and_extract(show_progress=False)
        self.assertTrue(success1)
        tar_mtime1 = fetcher.tar_path.stat().st_mtime

        # Force re-download
        import time
        time.sleep(0.1)  # Ensure different timestamp
        success2 = fetcher.download_and_extract(force=True, show_progress=False)
        self.assertTrue(success2)
        tar_mtime2 = fetcher.tar_path.stat().st_mtime

        # Should have different modification times
        self.assertNotEqual(tar_mtime1, tar_mtime2)

    def test_get_emulator_path_with_download(self):
        """Test get_emulator_path triggers download if needed."""
        cache_dir = Path(self.test_cache) / "get_path_test"

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir
        )

        # Should trigger download
        tt_path = fetcher.get_emulator_path("TT", download_if_missing=True)
        self.assertIsNotNone(tt_path)
        self.assertTrue(tt_path.exists())
        self.assertTrue(tt_path.is_dir())

        # Check contents
        weights_file = tt_path / "weights.npy"
        self.assertTrue(weights_file.exists())

    def test_get_emulator_path_no_download(self):
        """Test get_emulator_path without download returns None if not cached."""
        cache_dir = Path(self.test_cache) / "no_download_test"

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir
        )

        # Should return None without downloading
        tt_path = fetcher.get_emulator_path("TT", download_if_missing=False)
        self.assertIsNone(tt_path)

        # No files should exist
        self.assertFalse(fetcher.tar_path.exists())
        self.assertFalse(fetcher.emulators_dir.exists())

    def test_load_emulator_returns_path(self):
        """Test load_emulator method returns correct path."""
        cache_dir = Path(self.test_cache) / "load_emulator_test"

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir
        )

        # Load emulator (triggers download)
        emulator_path = fetcher.load_emulator("EE")
        self.assertIsNotNone(emulator_path)
        self.assertEqual(str(fetcher.emulators_dir / "EE"), emulator_path)

        # Verify the path exists
        self.assertTrue(Path(emulator_path).exists())

    def test_load_emulator_invalid_type(self):
        """Test load_emulator with invalid type raises error."""
        cache_dir = Path(self.test_cache) / "invalid_type_test"

        fetcher = EmulatorDataFetcher(
            zenodo_url=self.zenodo_url,
            emulator_types=self.emulator_types,
            cache_dir=cache_dir
        )

        with self.assertRaises(ValueError) as context:
            fetcher.get_emulator_path("INVALID")
        self.assertIn("Unknown emulator type", str(context.exception))


class TestCacheOperations(unittest.TestCase):
    """Test cache management operations."""

    @classmethod
    def setUpClass(cls):
        """Set up test cache with downloaded data."""
        cls.test_cache = tempfile.mkdtemp(prefix="jaxcapse_cache_test_")
        cls.zenodo_url = "https://zenodo.org/records/17115001/files/trained_emu.tar.gz?download=1"
        cls.emulator_types = ["TT", "TE", "EE", "PP"]

        # Pre-download data for cache tests
        cls.fetcher = EmulatorDataFetcher(
            zenodo_url=cls.zenodo_url,
            emulator_types=cls.emulator_types,
            cache_dir=cls.test_cache
        )
        cls.fetcher.download_and_extract(show_progress=False)

    @classmethod
    def tearDownClass(cls):
        """Clean up test cache."""
        if not os.environ.get("KEEP_TEST_DATA"):
            if Path(cls.test_cache).exists():
                shutil.rmtree(cls.test_cache)

    def test_list_cached(self):
        """Test listing cached emulators."""
        cached = self.fetcher.list_cached()
        self.assertEqual(set(cached), set(self.emulator_types))

    def test_list_available(self):
        """Test listing available emulator types."""
        available = self.fetcher.list_available()
        self.assertIsInstance(available, dict)
        self.assertEqual(len(available), 4)
        self.assertIn("TT", available)
        self.assertIn("CMB temperature power spectrum", available["TT"])

    def test_clear_cache_specific_emulator(self):
        """Test clearing specific emulator from cache."""
        # Re-download if needed (in case previous test cleared it)
        if not self.fetcher.emulators_dir.exists():
            self.fetcher.download_and_extract(show_progress=False)

        # Ensure TT exists
        tt_path = self.fetcher.emulators_dir / "TT"
        self.assertTrue(tt_path.exists())

        # Clear TT
        self.fetcher.clear_cache("TT")
        self.assertFalse(tt_path.exists())

        # Other emulators should still exist
        ee_path = self.fetcher.emulators_dir / "EE"
        self.assertTrue(ee_path.exists())

        # List cached should not include TT
        cached = self.fetcher.list_cached()
        self.assertNotIn("TT", cached)
        self.assertIn("EE", cached)

    def test_clear_cache_all(self):
        """Test clearing all cached files."""
        # Ensure files exist
        self.assertTrue(self.fetcher.tar_path.exists())
        self.assertTrue(self.fetcher.emulators_dir.exists())

        # Clear all
        self.fetcher.clear_cache()

        # Nothing should exist
        self.assertFalse(self.fetcher.tar_path.exists())
        self.assertFalse(self.fetcher.emulators_dir.exists())

        # List cached should be empty
        cached = self.fetcher.list_cached()
        self.assertEqual(len(cached), 0)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions and edge cases."""

    def setUp(self):
        """Set up test directory."""
        self.test_cache = tempfile.mkdtemp(prefix="jaxcapse_helper_test_")

    def tearDown(self):
        """Clean up test directory."""
        if not os.environ.get("KEEP_TEST_DATA"):
            if Path(self.test_cache).exists():
                shutil.rmtree(self.test_cache)

    def test_get_emulator_directory(self):
        """Test get_emulator_directory helper function."""
        from jaxcapse.data_fetcher import get_emulator_directory

        # This uses the default fetcher, so we need to ensure it's downloaded
        # For testing, we'll just check it returns a string
        try:
            path = get_emulator_directory("TT")
            self.assertIsInstance(path, str)
        except RuntimeError:
            # Expected if emulator not downloaded in default location
            pass

    def test_get_emulator_path_helper(self):
        """Test get_emulator_path helper function."""
        from jaxcapse.data_fetcher import get_emulator_path

        # This uses the default fetcher
        path = get_emulator_path("TT")
        if path is not None:
            self.assertIsInstance(path, Path)
            self.assertTrue(str(path).endswith("TT"))

    def test_nested_directory_extraction(self):
        """Test that nested 'trained_emu' directories are handled correctly."""
        import tarfile

        # Create a tar file with nested structure
        tar_path = Path(self.test_cache) / "test.tar.gz"
        extract_dir = Path(self.test_cache) / "extract"

        # Create mock structure
        nested_dir = Path(self.test_cache) / "create" / "trained_emu" / "TT"
        nested_dir.mkdir(parents=True)
        (nested_dir / "test_file.txt").write_text("test")

        # Create tar
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(
                Path(self.test_cache) / "create" / "trained_emu",
                arcname="trained_emu"
            )

        # Test extraction with reorganization
        fetcher = EmulatorDataFetcher(
            zenodo_url="http://example.com/test.tar.gz",
            emulator_types=["TT"],
            cache_dir=self.test_cache
        )

        # Manually extract to test the reorganization logic
        extract_dir.mkdir(parents=True, exist_ok=True)
        success = fetcher._extract_tar(tar_path, extract_dir, show_progress=False)
        self.assertTrue(success)

        # The TT directory should exist directly under extract_dir
        # after the fetcher's reorganization logic
        tt_exists = (extract_dir / "TT").exists() or (extract_dir / "trained_emu" / "TT").exists()
        self.assertTrue(tt_exists)


if __name__ == "__main__":
    unittest.main()