"""
Integration tests for the TTS pipeline (config -> TTS -> audio).
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

from justsayit.config.schema import AppConfig, TTSConfig
from justsayit.config.loader import ConfigLoader
from justsayit.core.tts_engine import TTSEngine, AudioData
from justsayit.core.audio_manager import AudioManager, PlaybackState
from justsayit.core.text_capture import TextCapture
from justsayit.utils.models import ModelManager


class TestTTSPipeline:
    """Test the complete TTS pipeline integration."""
    
    @pytest.fixture
    def temp_config(self):
        """Create a temporary configuration for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.yaml"
            yield config_path
    
    @pytest.fixture
    def config_loader(self, temp_config):
        """Create a config loader with temporary config."""
        return ConfigLoader(temp_config)
    
    @pytest.fixture
    def mock_model_manager(self):
        """Create a mock model manager for testing."""
        manager = Mock(spec=ModelManager)
        
        # Mock successful operations
        manager.is_model_downloaded.return_value = True
        manager.download_model.return_value = True
        
        # Mock model loading - return a simple mock model
        mock_model = {
            "processor": Mock(),
            "model": Mock(), 
            "device": "cpu"
        }
        manager.load_model.return_value = mock_model
        
        return manager
    
    @pytest.fixture
    def tts_engine(self, config_loader, mock_model_manager):
        """Create TTS engine with mocked dependencies."""
        config = config_loader.load_config()
        return TTSEngine(config.settings.tts, mock_model_manager)
    
    @pytest.fixture
    def audio_manager(self):
        """Create audio manager for testing.""" 
        return AudioManager()
    
    def test_config_to_tts_integration(self, config_loader, mock_model_manager):
        """Test configuration loading and TTS engine initialization."""
        # Load config
        config = config_loader.load_config()
        assert isinstance(config, AppConfig)
        
        # Create TTS engine with config
        tts_engine = TTSEngine(config.settings.tts, mock_model_manager)
        
        # Verify TTS engine uses config values
        assert tts_engine.config.speech_rate == config.settings.tts.speech_rate
        assert tts_engine.config.volume == config.settings.tts.volume
        assert tts_engine.config.pitch == config.settings.tts.pitch
    
    @patch('justsayit.core.tts_engine.torch.zeros')
    @patch('justsayit.core.tts_engine.torch.no_grad')
    def test_tts_synthesis_mock(self, mock_no_grad, mock_zeros, tts_engine):
        """Test TTS synthesis with mocked PyTorch operations."""
        # Mock torch operations
        mock_zeros.return_value.to.return_value = Mock()
        mock_no_grad.return_value.__enter__ = Mock()
        mock_no_grad.return_value.__exit__ = Mock()
        
        # Mock the model's generate_speech method
        mock_speech = Mock()
        mock_speech.cpu.return_value.numpy.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Load model (mocked)
        success = tts_engine.load_model("speecht5")
        assert success
        
        # Test synthesis would work if we had real models
        # For now, just test the engine is properly configured
        assert tts_engine.is_model_loaded()
        assert tts_engine.get_current_model() == "speecht5"
    
    def test_audio_manager_basic_operations(self, audio_manager):
        """Test basic audio manager operations."""
        # Test initial state
        assert audio_manager.get_playback_state() == PlaybackState.STOPPED
        assert not audio_manager.is_playing()
        assert audio_manager.get_queue_size() == 0
        
        # Test volume control
        audio_manager.set_volume(0.5)
        assert audio_manager.get_volume() == 0.5
        
        # Test device querying
        devices = audio_manager.get_output_devices()
        assert isinstance(devices, list)
        # Should have at least default device on most systems
    
    def test_audio_data_creation(self):
        """Test AudioData creation and properties."""
        import numpy as np
        
        # Create test audio data
        sample_rate = 22050
        duration = 2.0  # seconds
        samples = int(sample_rate * duration)
        
        audio_array = np.random.rand(samples) * 0.1  # Quiet random audio
        audio_data = AudioData(data=audio_array, sample_rate=sample_rate)
        
        # Test properties
        assert audio_data.sample_rate == sample_rate
        assert len(audio_data.data) == samples
        assert abs(audio_data.duration() - duration) < 0.01  # Close to expected duration
    
    def test_text_capture_basic(self):
        """Test basic text capture functionality."""
        # Note: This test requires a QApplication to be running
        # In practice, this would be handled by the main application
        
        text_capture = TextCapture()
        
        # Test getting text (may return None if no clipboard text)
        text = text_capture.get_selected_text()
        assert text is None or isinstance(text, str)
        
        # Test cleanup
        text_capture.cleanup()
    
    def test_config_parameter_validation(self, config_loader):
        """Test configuration parameter validation affects TTS."""
        config = config_loader.load_config()
        
        # Test valid parameter changes
        config.settings.tts.speech_rate = 1.5
        config.settings.tts.volume = 0.7
        config.settings.tts.pitch = 1.2
        
        # Save and reload
        success = config_loader.save_config(config)
        assert success
        
        reloaded_config = config_loader.load_config()
        assert reloaded_config.settings.tts.speech_rate == 1.5
        assert reloaded_config.settings.tts.volume == 0.7
        assert reloaded_config.settings.tts.pitch == 1.2
    
    def test_voice_parameter_updates(self, tts_engine):
        """Test voice parameter updates on TTS engine."""
        # Test parameter setting
        tts_engine.set_voice_parameters(
            speech_rate=1.2,
            pitch=0.9,
            volume=0.6
        )
        
        assert tts_engine.config.speech_rate == 1.2
        assert tts_engine.config.pitch == 0.9
        assert tts_engine.config.volume == 0.6
        
        # Test boundary clamping
        tts_engine.set_voice_parameters(
            speech_rate=5.0,  # Should clamp to 3.0
            volume=-0.5  # Should clamp to 0.0
        )
        
        assert tts_engine.config.speech_rate == 3.0
        assert tts_engine.config.volume == 0.0
    
    def test_model_manager_integration(self, mock_model_manager):
        """Test model manager integration with TTS engine."""
        config = TTSConfig()
        tts_engine = TTSEngine(config, mock_model_manager)
        
        # Test model loading
        success = tts_engine.load_model("speecht5")
        assert success
        
        # Verify model manager was called correctly
        mock_model_manager.is_model_downloaded.assert_called_with("speecht5", "tts")
        mock_model_manager.load_model.assert_called_with("speecht5", "tts", tts_engine.device)
        
        # Test model unloading
        tts_engine.unload_model()
        mock_model_manager.unload_model.assert_called_with("speecht5", "tts")
    
    def test_error_handling(self, config_loader):
        """Test error handling in the pipeline."""
        # Test with invalid model manager
        config = config_loader.load_config()
        
        # Create TTS engine with None model manager
        tts_engine = TTSEngine(config.settings.tts, None)
        
        # Should handle gracefully
        assert not tts_engine.is_model_loaded()
    
    def test_cleanup_operations(self, tts_engine, audio_manager):
        """Test cleanup operations work properly."""
        # Test TTS cleanup
        tts_engine.cleanup()
        assert not tts_engine.is_model_loaded()
        
        # Test audio manager cleanup
        initial_state = audio_manager.get_playback_state()
        audio_manager.cleanup()
        # Should be stopped after cleanup
        assert audio_manager.get_playback_state() == PlaybackState.STOPPED