"""
Unit tests for configuration system.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from justsayit.config.schema import (
    AppConfig, HotkeyConfig, TTSConfig, OCRConfig, UIConfig, SystemConfig,
    SettingsConfig, get_config_dir, get_default_config_path
)
from justsayit.config.loader import ConfigLoader


class TestConfigSchema:
    """Test configuration schema validation."""
    
    def test_default_config_creation(self):
        """Test creating default configuration."""
        config = AppConfig()
        
        assert config.hotkeys.tts_selection == "ctrl+shift+s"
        assert config.settings.tts.speech_rate == 1.0
        assert config.settings.tts.volume == 0.8
        assert config.settings.system.log_level == "INFO"
    
    def test_hotkey_validation_valid(self):
        """Test valid hotkey formats."""
        valid_hotkeys = [
            "ctrl+shift+s",
            "alt+f4",
            "cmd+space",
            "super+shift+a"
        ]
        
        for hotkey in valid_hotkeys:
            config = HotkeyConfig(tts_selection=hotkey)
            assert config.tts_selection == hotkey
    
    def test_hotkey_validation_invalid(self):
        """Test invalid hotkey formats."""
        invalid_hotkeys = [
            "",
            "s",  # No modifier
            "ctrl+",  # No key
            "invalid+s",  # Invalid modifier
            "ctrl+shift+",  # No key after modifiers
        ]
        
        for hotkey in invalid_hotkeys:
            with pytest.raises(ValueError):
                HotkeyConfig(tts_selection=hotkey)
    
    def test_tts_config_validation(self):
        """Test TTS configuration validation."""
        # Valid values
        config = TTSConfig(speech_rate=1.5, pitch=0.8, volume=0.9)
        assert config.speech_rate == 1.5
        assert config.pitch == 0.8
        assert config.volume == 0.9
        
        # Invalid values
        with pytest.raises(ValueError):
            TTSConfig(speech_rate=0.05)  # Too low
            
        with pytest.raises(ValueError):
            TTSConfig(speech_rate=4.0)  # Too high
            
        with pytest.raises(ValueError):
            TTSConfig(volume=1.5)  # Too high
    
    def test_system_config_validation(self):
        """Test system configuration validation."""
        # Valid log level
        config = SystemConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"
        
        # Invalid log level
        with pytest.raises(ValueError):
            SystemConfig(log_level="INVALID")
        
        # Case insensitive
        config = SystemConfig(log_level="debug")
        assert config.log_level == "DEBUG"
    
    def test_ui_config_validation(self):
        """Test UI configuration validation."""
        # Valid theme
        config = UIConfig(theme="dark")
        assert config.theme == "dark"
        
        # Invalid theme
        with pytest.raises(ValueError):
            UIConfig(theme="rainbow")
    
    def test_config_serialization(self):
        """Test configuration serialization to dict."""
        config = AppConfig()
        config_dict = config.dict()
        
        assert "hotkeys" in config_dict
        assert "settings" in config_dict
        assert config_dict["hotkeys"]["tts_selection"] == "ctrl+shift+s"


class TestConfigLoader:
    """Test configuration loader functionality."""
    
    def test_load_default_config_no_file(self):
        """Test loading default config when no file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            loader = ConfigLoader(config_path)
            
            config = loader.load_config()
            
            assert isinstance(config, AppConfig)
            assert config.hotkeys.tts_selection == "ctrl+shift+s"
            assert config_path.exists()  # Should create default file
    
    def test_load_existing_config(self):
        """Test loading existing configuration file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Create a test config file
            test_config = {
                "hotkeys": {
                    "tts_selection": "alt+f1",
                    "ocr_area": "alt+f2",
                    "stop_speech": "alt+f3"
                },
                "settings": {
                    "tts": {
                        "speech_rate": 2.0,
                        "volume": 0.5
                    }
                }
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(test_config, f)
            
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            assert config.hotkeys.tts_selection == "alt+f1"
            assert config.settings.tts.speech_rate == 2.0
            assert config.settings.tts.volume == 0.5
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            loader = ConfigLoader(config_path)
            
            # Create modified config
            config = AppConfig()
            config.hotkeys.tts_selection = "ctrl+alt+s"
            config.settings.tts.speech_rate = 1.5
            
            # Save config
            success = loader.save_config(config)
            assert success
            assert config_path.exists()
            
            # Verify saved content
            loader2 = ConfigLoader(config_path)
            loaded_config = loader2.load_config()
            
            assert loaded_config.hotkeys.tts_selection == "ctrl+alt+s"
            assert loaded_config.settings.tts.speech_rate == 1.5
    
    def test_invalid_yaml_handling(self):
        """Test handling of invalid YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            
            # Write invalid YAML
            with open(config_path, 'w') as f:
                f.write("invalid: yaml: content: [unclosed")
            
            loader = ConfigLoader(config_path)
            config = loader.load_config()
            
            # Should fall back to default config
            assert isinstance(config, AppConfig)
            assert config.hotkeys.tts_selection == "ctrl+shift+s"
    
    def test_config_change_callbacks(self):
        """Test configuration change callbacks."""
        callback_mock = MagicMock()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            loader = ConfigLoader(config_path)
            
            # Load initial config
            old_config = loader.load_config()
            
            # Add callback
            loader.add_change_callback(callback_mock)
            
            # Trigger reload
            loader.reload_config()
            
            # Verify callback was called
            callback_mock.assert_called_once()
            args = callback_mock.call_args[0]
            assert len(args) == 2  # old_config, new_config
    
    @patch('platform.system')
    def test_config_dir_platforms(self, mock_system):
        """Test configuration directory paths on different platforms."""
        # Test macOS
        mock_system.return_value = "Darwin"
        config_dir = get_config_dir()
        assert str(config_dir).endswith("/.justsayit")
        
        # Test Windows
        mock_system.return_value = "Windows"
        config_dir = get_config_dir()
        assert str(config_dir).endswith(".justsayit")
        
        # Test Linux
        mock_system.return_value = "Linux"
        config_dir = get_config_dir()
        assert str(config_dir).endswith("/.justsayit")