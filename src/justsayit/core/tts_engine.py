"""
Text-to-Speech engine with support for multiple models and voice customization.
"""

import logging
import io
import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass

from ..utils.models import ModelManager
from ..utils.platform import get_optimal_device
from ..config.schema import TTSConfig


logger = logging.getLogger(__name__)


@dataclass
class AudioData:
    """Container for audio data."""
    data: np.ndarray
    sample_rate: int
    channels: int = 1
    
    def duration(self) -> float:
        """Get audio duration in seconds."""
        return len(self.data) / self.sample_rate


class TTSEngine:
    """Text-to-Speech engine with model management and voice control."""
    
    def __init__(self, config: TTSConfig, model_manager: Optional[ModelManager] = None):
        self.config = config
        self.model_manager = model_manager or ModelManager()
        self.device = get_optimal_device()
        
        self._current_model = None
        self._current_model_name = None
        self._model_objects = None
        
        logger.info(f"Initialized TTS engine with device: {self.device}")
    
    def load_model(self, model_name: Optional[str] = None) -> bool:
        """Load a TTS model for inference."""
        if model_name is None:
            # Extract model name from config (e.g., "coqui-tts/model" -> "coqui-fast")
            model_name = self._extract_model_name(self.config.model)
        
        if self._current_model_name == model_name and self._model_objects is not None:
            logger.debug(f"Model {model_name} already loaded")
            return True
        
        try:
            # Download model if not available
            if not self.model_manager.is_model_downloaded(model_name, "tts"):
                logger.info(f"Model {model_name} not found, downloading...")
                success = self.model_manager.download_model(model_name, "tts")
                if not success:
                    logger.error(f"Failed to download model {model_name}")
                    return False
            
            # Load the model
            self._model_objects = self.model_manager.load_model(model_name, "tts", self.device)
            if self._model_objects is None:
                logger.error(f"Failed to load model {model_name}")
                return False
            
            self._current_model_name = model_name
            logger.info(f"Successfully loaded TTS model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading TTS model {model_name}: {e}")
            return False
    
    def synthesize(self, text: str) -> Optional[AudioData]:
        """Convert text to speech and return audio data."""
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None
        
        if self._model_objects is None:
            logger.error("No model loaded for synthesis")
            return None
        
        try:
            text = self._preprocess_text(text)
            logger.debug(f"Synthesizing text: {text[:50]}...")
            
            # Generate audio based on model type
            if self._current_model_name == "speecht5":
                audio_data = self._synthesize_speecht5(text)
            elif self._current_model_name == "coqui-fast":
                audio_data = self._synthesize_coqui(text)
            elif self._current_model_name == "bark-small":
                audio_data = self._synthesize_bark(text)
            else:
                raise ValueError(f"Unknown model: {self._current_model_name}")
            
            if audio_data is not None:
                # Apply voice parameters (speed, pitch, volume)
                audio_data = self._apply_voice_parameters(audio_data)
                logger.debug(f"Generated audio: {audio_data.duration():.2f}s")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error during synthesis: {e}")
            return None
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices for the current model."""
        if self._current_model_name == "speecht5":
            return ["default", "female", "male"]
        elif self._current_model_name == "coqui-fast":
            return ["default", "female", "male", "neutral"]
        elif self._current_model_name == "bark-small":
            return ["v2/en_speaker_0", "v2/en_speaker_1", "v2/en_speaker_2", 
                   "v2/en_speaker_3", "v2/en_speaker_4", "v2/en_speaker_5"]
        else:
            return ["default"]
    
    def set_voice_parameters(self, speech_rate: Optional[float] = None,
                           pitch: Optional[float] = None,
                           volume: Optional[float] = None):
        """Update voice parameters."""
        if speech_rate is not None:
            self.config.speech_rate = max(0.1, min(3.0, speech_rate))
        if pitch is not None:
            self.config.pitch = max(0.1, min(3.0, pitch))
        if volume is not None:
            self.config.volume = max(0.0, min(1.0, volume))
        
        logger.debug(f"Updated voice parameters: rate={self.config.speech_rate}, "
                    f"pitch={self.config.pitch}, volume={self.config.volume}")
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model_objects is not None
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently loaded model name."""
        return self._current_model_name
    
    def unload_model(self):
        """Unload the current model to free memory."""
        if self._current_model_name:
            self.model_manager.unload_model(self._current_model_name, "tts")
            self._model_objects = None
            self._current_model_name = None
            logger.info("Unloaded TTS model")
    
    def _extract_model_name(self, model_path: str) -> str:
        """Extract simplified model name from config path."""
        # Map common model paths to our simplified names
        model_mappings = {
            "microsoft/speecht5_tts": "speecht5",
            "coqui/XTTS-v2": "coqui-fast", 
            "suno/bark": "bark-small",
            "coqui-tts/tts_models/en/ljspeech/tacotron2-DDC": "coqui-fast"
        }
        
        for path, name in model_mappings.items():
            if path in model_path:
                return name
        
        # Default to speecht5 for unknown models
        logger.warning(f"Unknown model path {model_path}, defaulting to speecht5")
        return "speecht5"
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better TTS quality."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Expand common abbreviations
        abbreviations = {
            "Dr.": "Doctor",
            "Mr.": "Mister", 
            "Mrs.": "Missus",
            "Ms.": "Miss",
            "etc.": "etcetera",
            "vs.": "versus",
            "e.g.": "for example",
            "i.e.": "that is"
        }
        
        for abbr, expansion in abbreviations.items():
            text = text.replace(abbr, expansion)
        
        # Limit length for better processing
        if len(text) > 1000:
            text = text[:1000] + "..."
            logger.warning("Text truncated to 1000 characters")
        
        return text
    
    def _synthesize_speecht5(self, text: str) -> Optional[AudioData]:
        """Synthesize audio using SpeechT5 model."""
        try:
            from transformers import SpeechT5HifiGan
            
            processor = self._model_objects["processor"]
            model = self._model_objects["model"]
            
            # Load vocoder if not already loaded
            if "vocoder" not in self._model_objects:
                vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
                vocoder.to(self.device)
                self._model_objects["vocoder"] = vocoder
            
            vocoder = self._model_objects["vocoder"]
            
            # Process input text
            inputs = processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate speaker embeddings (default voice)
            speaker_embeddings = torch.zeros((1, 512)).to(self.device)
            
            # Generate speech
            with torch.no_grad():
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            
            # Convert to numpy
            audio_array = speech.cpu().numpy()
            sample_rate = 16000  # SpeechT5 default
            
            return AudioData(data=audio_array, sample_rate=sample_rate)
            
        except Exception as e:
            logger.error(f"Error in SpeechT5 synthesis: {e}")
            return None
    
    def _synthesize_coqui(self, text: str) -> Optional[AudioData]:
        """Synthesize audio using Coqui TTS model."""
        try:
            model = self._model_objects["model"]
            
            # Generate speech with Coqui TTS
            audio_array = model.tts(text)
            sample_rate = 22050  # Coqui default
            
            return AudioData(data=np.array(audio_array), sample_rate=sample_rate)
            
        except Exception as e:
            logger.error(f"Error in Coqui synthesis: {e}")
            return None
    
    def _synthesize_bark(self, text: str) -> Optional[AudioData]:
        """Synthesize audio using Bark model."""
        try:
            import bark
            from bark import SAMPLE_RATE
            
            # Generate speech with Bark
            audio_array = bark.text_to_audio(text, history_prompt=self.config.voice)
            
            return AudioData(data=audio_array, sample_rate=SAMPLE_RATE)
            
        except Exception as e:
            logger.error(f"Error in Bark synthesis: {e}")
            return None
    
    def _apply_voice_parameters(self, audio_data: AudioData) -> AudioData:
        """Apply speech rate, pitch, and volume modifications."""
        try:
            import librosa
            
            audio = audio_data.data.copy()
            sample_rate = audio_data.sample_rate
            
            # Apply speed change
            if self.config.speech_rate != 1.0:
                audio = librosa.effects.time_stretch(audio, rate=self.config.speech_rate)
            
            # Apply pitch change 
            if self.config.pitch != 1.0:
                # Convert pitch multiplier to semitones
                n_steps = 12 * np.log2(self.config.pitch)
                audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)
            
            # Apply volume change
            if self.config.volume != 1.0:
                audio = audio * self.config.volume
                # Prevent clipping
                audio = np.clip(audio, -1.0, 1.0)
            
            return AudioData(data=audio, sample_rate=sample_rate, channels=audio_data.channels)
            
        except Exception as e:
            logger.error(f"Error applying voice parameters: {e}")
            return audio_data  # Return original if processing fails
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up TTS engine...")
        self.unload_model()
        if self.model_manager:
            self.model_manager.cleanup()