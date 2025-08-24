"""
Model management utilities for downloading, caching, and loading ML models.
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import requests
from huggingface_hub import hf_hub_download, list_repo_files
import torch

from .platform import get_models_dir


logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    repo_id: str
    files: List[str]
    size_mb: Optional[int] = None
    description: Optional[str] = None
    requirements: Optional[List[str]] = None


class ModelManager:
    """Manages downloading, caching, and loading of ML models."""
    
    # Available TTS models with their configurations
    AVAILABLE_TTS_MODELS = {
        "coqui-fast": ModelInfo(
            name="coqui-fast",
            repo_id="coqui/XTTS-v2",
            files=["model.pth", "config.json", "vocab.json"],
            size_mb=1800,
            description="Fast, high-quality multilingual TTS model",
            requirements=["TTS>=0.17.0"]
        ),
        "speecht5": ModelInfo(
            name="speecht5",
            repo_id="microsoft/speecht5_tts",
            files=["pytorch_model.bin", "config.json"],
            size_mb=400,
            description="Microsoft's SpeechT5 TTS model - lightweight and fast",
            requirements=["transformers>=4.30.0", "speechbrain>=0.5.0"]
        ),
        "bark-small": ModelInfo(
            name="bark-small",
            repo_id="suno/bark",
            files=["pytorch_model.bin", "config.json"],
            size_mb=2400,
            description="Suno's Bark model - very natural but resource intensive",
            requirements=["bark>=1.0.0"]
        )
    }
    
    # Available OCR models
    AVAILABLE_OCR_MODELS = {
        "trocr-base": ModelInfo(
            name="trocr-base",
            repo_id="microsoft/trocr-base-printed",
            files=["pytorch_model.bin", "config.json", "preprocessor_config.json"],
            size_mb=558,
            description="Microsoft's TrOCR base model for printed text",
            requirements=["transformers>=4.30.0"]
        ),
        "easyocr": ModelInfo(
            name="easyocr",
            repo_id="easyocr/easyocr",
            files=["craft_mlt_25k.pth", "CRNN_VGG_BiLSTM_CTC.pth"],
            size_mb=150,
            description="EasyOCR - fast and accurate OCR",
            requirements=["easyocr>=1.7.0"]
        )
    }
    
    def __init__(self):
        self.models_dir = get_models_dir()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models = {}  # Cache for loaded models
    
    def get_available_tts_models(self) -> Dict[str, ModelInfo]:
        """Get list of available TTS models."""
        return self.AVAILABLE_TTS_MODELS.copy()
    
    def get_available_ocr_models(self) -> Dict[str, ModelInfo]:
        """Get list of available OCR models."""
        return self.AVAILABLE_OCR_MODELS.copy()
    
    def is_model_downloaded(self, model_name: str, model_type: str = "tts") -> bool:
        """Check if a model is already downloaded."""
        model_info = self._get_model_info(model_name, model_type)
        if not model_info:
            return False
            
        model_dir = self.models_dir / model_type / model_name
        if not model_dir.exists():
            return False
            
        # Check if all required files exist
        for file_name in model_info.files:
            if not (model_dir / file_name).exists():
                return False
                
        return True
    
    def download_model(self, model_name: str, model_type: str = "tts", 
                      progress_callback: Optional[callable] = None) -> bool:
        """Download a model from Hugging Face Hub."""
        model_info = self._get_model_info(model_name, model_type)
        if not model_info:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        model_dir = self.models_dir / model_type / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading model {model_name} from {model_info.repo_id}")
            
            total_files = len(model_info.files)
            for i, file_name in enumerate(model_info.files):
                if progress_callback:
                    progress_callback(i / total_files, f"Downloading {file_name}")
                
                # Download file from Hugging Face Hub
                downloaded_path = hf_hub_download(
                    repo_id=model_info.repo_id,
                    filename=file_name,
                    cache_dir=str(model_dir),
                    local_files_only=False
                )
                
                # Move to our model directory if needed
                target_path = model_dir / file_name
                if Path(downloaded_path) != target_path:
                    import shutil
                    shutil.move(downloaded_path, target_path)
            
            if progress_callback:
                progress_callback(1.0, f"Model {model_name} downloaded successfully")
                
            logger.info(f"Successfully downloaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
            # Clean up partial download
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
            return False
    
    def load_model(self, model_name: str, model_type: str = "tts", 
                   device: Optional[str] = None) -> Optional[Any]:
        """Load a model into memory."""
        cache_key = f"{model_type}_{model_name}"
        
        # Return cached model if available
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        if not self.is_model_downloaded(model_name, model_type):
            logger.error(f"Model {model_name} is not downloaded")
            return None
        
        try:
            model_dir = self.models_dir / model_type / model_name
            
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading model {model_name} on device {device}")
            
            # Load model based on type
            if model_type == "tts":
                model = self._load_tts_model(model_name, model_dir, device)
            elif model_type == "ocr":
                model = self._load_ocr_model(model_name, model_dir, device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Cache the loaded model
            self._loaded_models[cache_key] = model
            
            logger.info(f"Successfully loaded model {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    def unload_model(self, model_name: str, model_type: str = "tts"):
        """Unload a model from memory."""
        cache_key = f"{model_type}_{model_name}"
        if cache_key in self._loaded_models:
            del self._loaded_models[cache_key]
            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Unloaded model {model_name}")
    
    def get_model_size(self, model_name: str, model_type: str = "tts") -> int:
        """Get the size of a downloaded model in bytes."""
        if not self.is_model_downloaded(model_name, model_type):
            return 0
            
        model_dir = self.models_dir / model_type / model_name
        total_size = 0
        
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                
        return total_size
    
    def delete_model(self, model_name: str, model_type: str = "tts") -> bool:
        """Delete a downloaded model."""
        try:
            # Unload from memory first
            self.unload_model(model_name, model_type)
            
            model_dir = self.models_dir / model_type / model_name
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                logger.info(f"Deleted model {model_name}")
                return True
            else:
                logger.warning(f"Model {model_name} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def _get_model_info(self, model_name: str, model_type: str) -> Optional[ModelInfo]:
        """Get model info from the registry."""
        if model_type == "tts":
            return self.AVAILABLE_TTS_MODELS.get(model_name)
        elif model_type == "ocr":
            return self.AVAILABLE_OCR_MODELS.get(model_name)
        return None
    
    def _load_tts_model(self, model_name: str, model_dir: Path, device: str) -> Any:
        """Load a TTS model."""
        if model_name == "speecht5":
            from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
            
            processor = SpeechT5Processor.from_pretrained(str(model_dir))
            model = SpeechT5ForTextToSpeech.from_pretrained(str(model_dir))
            model.to(device)
            
            return {"processor": processor, "model": model, "device": device}
            
        elif model_name == "coqui-fast":
            from TTS.api import TTS
            
            # Initialize Coqui TTS
            model = TTS(model_path=str(model_dir), progress_bar=False)
            return {"model": model, "device": device}
            
        elif model_name == "bark-small":
            import bark
            
            # Load Bark model
            model = bark.load_model(str(model_dir))
            return {"model": model, "device": device}
            
        else:
            raise ValueError(f"Unknown TTS model: {model_name}")
    
    def _load_ocr_model(self, model_name: str, model_dir: Path, device: str) -> Any:
        """Load an OCR model."""
        if model_name == "trocr-base":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            processor = TrOCRProcessor.from_pretrained(str(model_dir))
            model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
            model.to(device)
            
            return {"processor": processor, "model": model, "device": device}
            
        elif model_name == "easyocr":
            import easyocr
            
            # EasyOCR handles model loading internally
            reader = easyocr.Reader(['en'], model_storage_directory=str(model_dir))
            return {"reader": reader}
            
        else:
            raise ValueError(f"Unknown OCR model: {model_name}")
    
    def cleanup(self):
        """Clean up resources and unload all models."""
        logger.info("Cleaning up model manager...")
        model_names = list(self._loaded_models.keys())
        for cache_key in model_names:
            model_type, model_name = cache_key.split("_", 1)
            self.unload_model(model_name, model_type)