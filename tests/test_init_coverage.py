"""
Tests for __init__.py module to improve coverage.
Tests initialization, configuration, and error handling.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings
import shutil

# We need to test the import behavior, so we'll manipulate sys.modules


class TestInitializationCoverage(unittest.TestCase):
    """Tests for __init__.py module coverage."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        self.test_cache = tempfile.mkdtemp(prefix="jaxcapse_init_test_")

        # Remove jaxcapse from sys.modules to test fresh imports
        modules_to_remove = [key for key in sys.modules if key.startswith('jaxcapse')]
        for module in modules_to_remove:
            del sys.modules[module]

    def tearDown(self):
        """Restore environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

        if Path(self.test_cache).exists():
            shutil.rmtree(self.test_cache)

        # Clean up sys.modules again
        modules_to_remove = [key for key in sys.modules if key.startswith('jaxcapse')]
        for module in modules_to_remove:
            del sys.modules[module]

    def test_no_auto_download_environment(self):
        """Test initialization with JAXCAPSE_NO_AUTO_DOWNLOAD set."""
        # Set environment variable
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"

        # Import should not trigger download
        import jaxcapse

        # trained_emulators should have empty structure
        self.assertIn("camb_lcdm", jaxcapse.trained_emulators)
        self.assertIsNone(jaxcapse.trained_emulators["camb_lcdm"]["TT"])
        self.assertIsNone(jaxcapse.trained_emulators["camb_lcdm"]["EE"])

    def test_reload_emulators_function(self):
        """Test reload_emulators function."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Test reload all models
        with patch('jaxcapse._load_emulator_set') as mock_load:
            mock_load.return_value = {
                "TT": MagicMock(),
                "EE": MagicMock(),
                "TE": MagicMock(),
                "PP": MagicMock()
            }

            jaxcapse.reload_emulators()
            mock_load.assert_called()

            # Check that trained_emulators was updated
            self.assertIsNotNone(jaxcapse.trained_emulators["camb_lcdm"])

    def test_reload_specific_model(self):
        """Test reload_emulators for specific model."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Test reload specific model
        with patch('jaxcapse._load_emulator_set') as mock_load:
            mock_load.return_value = {"TT": MagicMock()}

            jaxcapse.reload_emulators("camb_lcdm")
            mock_load.assert_called_with(
                "camb_lcdm",
                jaxcapse.EMULATOR_CONFIGS["camb_lcdm"],
                auto_download=True
            )

    def test_reload_invalid_model(self):
        """Test reload_emulators with invalid model name."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Should raise ValueError for invalid model
        with self.assertRaises(ValueError) as context:
            jaxcapse.reload_emulators("invalid_model")

        self.assertIn("Unknown model", str(context.exception))

    def test_add_emulator_config_with_checksum(self):
        """Test adding new emulator configuration with checksum."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Add new configuration
        with patch('jaxcapse._load_emulator_set') as mock_load:
            mock_load.return_value = {"TT": MagicMock(), "EE": MagicMock()}

            result = jaxcapse.add_emulator_config(
                model_name="test_model",
                zenodo_url="https://example.com/test.tar.gz",
                emulator_types=["TT", "EE"],
                description="Test emulators",
                checksum="abc123",
                auto_load=True
            )

            # Check configuration was added
            self.assertIn("test_model", jaxcapse.EMULATOR_CONFIGS)
            self.assertEqual(
                jaxcapse.EMULATOR_CONFIGS["test_model"]["checksum"],
                "abc123"
            )
            self.assertEqual(
                jaxcapse.EMULATOR_CONFIGS["test_model"]["description"],
                "Test emulators"
            )

            # Check emulators were loaded
            mock_load.assert_called_once()
            self.assertIsNotNone(result)

    def test_add_emulator_config_no_auto_load(self):
        """Test adding configuration without auto-loading."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        result = jaxcapse.add_emulator_config(
            model_name="test_model_no_load",
            zenodo_url="https://example.com/test2.tar.gz",
            emulator_types=["TT"],
            auto_load=False
        )

        # Configuration should be added
        self.assertIn("test_model_no_load", jaxcapse.EMULATOR_CONFIGS)

        # Empty structure should be created
        self.assertIn("test_model_no_load", jaxcapse.trained_emulators)
        self.assertIsNone(jaxcapse.trained_emulators["test_model_no_load"]["TT"])

    def test_load_emulator_set_error_handling(self):
        """Test _load_emulator_set error handling."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Test with fetcher that fails
        with patch('jaxcapse.get_fetcher') as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.list_cached.return_value = []
            mock_fetcher.download_and_extract.side_effect = Exception("Download failed")
            mock_get_fetcher.return_value = mock_fetcher

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result = jaxcapse._load_emulator_set(
                    "test_model",
                    {"zenodo_url": "http://test.com", "emulator_types": ["TT"]},
                    auto_download=True
                )

                # Should return empty dict on failure
                self.assertEqual(result, {"TT": None})

                # Should have warned
                self.assertTrue(len(w) > 0)

    def test_load_emulator_set_partial_success(self):
        """Test _load_emulator_set with partial loading success."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse
        from jaxcapse import get_emulator_path, load_emulator

        config = {
            "zenodo_url": "http://test.com",
            "emulator_types": ["TT", "EE"],
            "checksum": "test123"
        }

        with patch('jaxcapse.get_fetcher') as mock_get_fetcher:
            mock_fetcher = MagicMock()
            mock_fetcher.list_cached.return_value = ["TT"]
            mock_fetcher.download_and_extract.return_value = True
            mock_get_fetcher.return_value = mock_fetcher

            # Mock get_emulator_path to return path for TT but not EE
            with patch('jaxcapse.get_emulator_path') as mock_get_path:
                tt_path = Path(self.test_cache) / "TT"
                tt_path.mkdir(parents=True)
                mock_get_path.side_effect = lambda x: tt_path if x == "TT" else None

                # Mock load_emulator
                with patch('jaxcapse.load_emulator') as mock_load:
                    mock_load.return_value = MagicMock()

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                        result = jaxcapse._load_emulator_set(
                            "test_model",
                            config,
                            auto_download=True
                        )

                        # TT should load, EE should be None
                        self.assertIsNotNone(result["TT"])
                        self.assertIsNone(result["EE"])

                        # Should have warned about EE
                        warning_messages = [str(warning.message) for warning in w]
                        self.assertTrue(
                            any("Could not find EE" in msg for msg in warning_messages)
                        )

    def test_initialization_with_download_failure(self):
        """Test initialization when download fails during import."""
        # Set environment variable to avoid actual download
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"

        # Clean modules
        modules_to_remove = [key for key in sys.modules if key.startswith('jaxcapse')]
        for module in modules_to_remove:
            del sys.modules[module]

        # Now test with simulated failure
        original_load = None

        def mock_load_with_failure(*args, **kwargs):
            if args[0] == "camb_lcdm":  # Only fail for our test case
                raise Exception("Simulated network error")
            return {}

        # Import fresh and patch
        import jaxcapse
        original_load = jaxcapse._load_emulator_set

        # Now test reload with failure
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            jaxcapse._load_emulator_set = mock_load_with_failure
            try:
                # This should trigger the error handling
                jaxcapse.reload_emulators("camb_lcdm")
            except Exception:
                pass  # Expected to fail
            finally:
                # Restore
                jaxcapse._load_emulator_set = original_load

            # Check structure still exists
            self.assertIn("camb_lcdm", jaxcapse.trained_emulators)

    def test_get_fetcher_with_checksum(self):
        """Test get_fetcher properly passes checksum."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse
        from jaxcapse.data_fetcher import get_fetcher

        # Test with custom checksum
        fetcher = get_fetcher(
            zenodo_url="https://example.com/test.tar.gz",
            emulator_types=["TT"],
            expected_checksum="custom_checksum_123"
        )

        self.assertEqual(fetcher.expected_checksum, "custom_checksum_123")

    def test_emulator_config_defaults(self):
        """Test EMULATOR_CONFIGS has correct default values."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Check default configuration
        self.assertIn("camb_lcdm", jaxcapse.EMULATOR_CONFIGS)
        config = jaxcapse.EMULATOR_CONFIGS["camb_lcdm"]

        self.assertIn("zenodo_url", config)
        self.assertIn("emulator_types", config)
        self.assertIn("description", config)
        self.assertIn("checksum", config)

        # Check checksum is the correct one
        self.assertEqual(
            config["checksum"],
            "b1d6f47c3bafb6b1ef0b80069e3d7982f274c6c7352ee44e460ffb9c2a573210"
        )

    def test_module_exports(self):
        """Test that __all__ exports are correct."""
        os.environ["JAXCAPSE_NO_AUTO_DOWNLOAD"] = "1"
        import jaxcapse

        # Check all exported names exist
        for name in jaxcapse.__all__:
            self.assertTrue(hasattr(jaxcapse, name), f"Missing export: {name}")

        # Check specific exports
        self.assertTrue(callable(jaxcapse.load_emulator))
        self.assertTrue(callable(jaxcapse.get_emulator_path))
        self.assertTrue(callable(jaxcapse.get_fetcher))
        self.assertTrue(callable(jaxcapse.add_emulator_config))
        self.assertTrue(callable(jaxcapse.reload_emulators))
        self.assertIsInstance(jaxcapse.trained_emulators, dict)
        self.assertIsInstance(jaxcapse.EMULATOR_CONFIGS, dict)


if __name__ == "__main__":
    unittest.main()