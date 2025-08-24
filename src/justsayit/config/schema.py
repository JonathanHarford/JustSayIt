"""
Configuration schema definitions using Pydantic for type safety and validation.
"""

from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator


class HotkeyConfig(BaseModel):
    """Configuration for application hotkeys."""
    
    tts_selection: str = Field(
        default="ctrl+shift+s",
        description="Hotkey to read selected text aloud"
    )
    ocr_area: str = Field(
        default="ctrl+shift+a", 
        description="Hotkey to start OCR area selection"
    )
    stop_speech: str = Field(
        default="ctrl+shift+x",
        description="Hotkey to stop current speech"
    )

    @validator('tts_selection', 'ocr_area', 'stop_speech')
    def validate_hotkey_format(cls, v):
        """Validate hotkey format (basic validation)."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Hotkey must be a non-empty string")
        
        # Basic format validation - should contain modifiers and key
        parts = v.lower().split('+')
        if len(parts) < 2:
            raise ValueError("Hotkey must contain at least one modifier and a key")
            
        valid_modifiers = {'ctrl', 'alt', 'shift', 'cmd', 'super', 'meta'}
        modifiers = parts[:-1]
        key = parts[-1]
        
        if not all(mod in valid_modifiers for mod in modifiers):
            raise ValueError(f"Invalid modifier. Valid modifiers: {valid_modifiers}")
            
        if not key or len(key) == 0:
            raise ValueError("Hotkey must end with a key")
            
        return v


class TTSConfig(BaseModel):
    """Configuration for Text-to-Speech engine."""
    
    model: str = Field(
        default="coqui-tts/tts_models/en/ljspeech/tacotron2-DDC",
        description="TTS model to use"
    )
    voice: str = Field(
        default="default",
        description="Voice to use for TTS"
    )
    speech_rate: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Speech rate multiplier (0.1 to 3.0)"
    )
    pitch: float = Field(
        default=1.0,
        ge=0.1,
        le=3.0,
        description="Pitch multiplier (0.1 to 3.0)"
    )
    volume: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Volume level (0.0 to 1.0)"
    )


class OCRConfig(BaseModel):
    """Configuration for Optical Character Recognition."""
    
    model: str = Field(
        default="microsoft/trocr-base-printed",
        description="OCR model to use"
    )
    preprocessing: bool = Field(
        default=True,
        description="Enable image preprocessing"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for text extraction"
    )


class UIConfig(BaseModel):
    """Configuration for user interface."""
    
    show_notifications: bool = Field(
        default=True,
        description="Show notifications for actions"
    )
    minimize_to_tray: bool = Field(
        default=True,
        description="Minimize to system tray instead of closing"
    )
    theme: str = Field(
        default="system",
        description="UI theme (light/dark/system)"
    )

    @validator('theme')
    def validate_theme(cls, v):
        """Validate theme selection."""
        valid_themes = {'light', 'dark', 'system'}
        if v not in valid_themes:
            raise ValueError(f"Theme must be one of: {valid_themes}")
        return v


class SystemConfig(BaseModel):
    """Configuration for system integration."""
    
    auto_start: bool = Field(
        default=False,
        description="Start JustSayIt automatically on system startup"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    max_log_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum log file size in MB"
    )

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class SettingsConfig(BaseModel):
    """Container for all settings."""
    
    tts: TTSConfig = Field(default_factory=TTSConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)


class AppConfig(BaseModel):
    """Root configuration model."""
    
    hotkeys: HotkeyConfig = Field(default_factory=HotkeyConfig)
    settings: SettingsConfig = Field(default_factory=SettingsConfig)

    class Config:
        """Pydantic configuration."""
        extra = 'forbid'  # Don't allow extra fields
        validate_assignment = True  # Validate on assignment


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    import platform
    
    system = platform.system()
    home = Path.home()
    
    if system == "Darwin":  # macOS
        config_dir = home / ".justsayit"
    elif system == "Windows":
        config_dir = Path.home() / ".justsayit"
    else:  # Linux and others
        # Follow XDG Base Directory spec
        xdg_config = Path.home() / ".config" / "justsayit"
        config_dir = Path.home() / ".justsayit"  # For simplicity, use same as others
    
    return config_dir


def get_default_config_path() -> Path:
    """Get the default configuration file path."""
    return get_config_dir() / "config.yaml"


def ensure_config_dir() -> Path:
    """Ensure configuration directory exists and return its path."""
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir