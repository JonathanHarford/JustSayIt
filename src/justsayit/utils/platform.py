"""
Platform-specific utilities and helpers.
"""

import platform
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


def get_platform() -> str:
    """Get the current platform name."""
    return platform.system()


def is_macos() -> bool:
    """Check if running on macOS."""
    return platform.system() == "Darwin"


def is_windows() -> bool:
    """Check if running on Windows."""  
    return platform.system() == "Windows"


def is_linux() -> bool:
    """Check if running on Linux."""
    return platform.system() == "Linux"


def get_models_dir() -> Path:
    """Get the directory for storing ML models."""
    if is_macos():
        return Path.home() / ".justsayit" / "models"
    elif is_windows():
        return Path.home() / ".justsayit" / "models"  
    else:  # Linux and others
        # Follow XDG Base Directory spec, but use simpler path for now
        return Path.home() / ".justsayit" / "models"


def get_cache_dir() -> Path:
    """Get the cache directory."""
    if is_macos():
        return Path.home() / "Library" / "Caches" / "JustSayIt"
    elif is_windows():
        import os
        cache_dir = os.environ.get("LOCALAPPDATA")
        if cache_dir:
            return Path(cache_dir) / "JustSayIt" / "Cache"
        return Path.home() / ".justsayit" / "cache"
    else:  # Linux
        cache_home = Path.home() / ".cache"
        return cache_home / "justsayit"


def get_log_dir() -> Path:
    """Get the directory for log files."""
    if is_macos():
        return Path.home() / "Library" / "Logs" / "JustSayIt"
    elif is_windows():
        import os
        local_data = os.environ.get("LOCALAPPDATA")
        if local_data:
            return Path(local_data) / "JustSayIt" / "Logs"
        return Path.home() / ".justsayit" / "logs"
    else:  # Linux
        return Path.home() / ".justsayit" / "logs"


def ensure_app_directories():
    """Ensure all application directories exist."""
    directories = [
        get_models_dir(),
        get_cache_dir(), 
        get_log_dir(),
        Path.home() / ".justsayit"  # Config directory
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")


def get_available_devices():
    """Get list of available compute devices (CPU, CUDA, MPS)."""
    devices = ["cpu"]
    
    try:
        import torch
        
        if torch.cuda.is_available():
            devices.append("cuda")
            # Add specific CUDA devices
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
        
        # Check for Apple Silicon MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            devices.append("mps")
            
    except ImportError:
        logger.debug("PyTorch not available, only CPU device will be used")
    
    return devices


def get_optimal_device() -> str:
    """Get the optimal device for ML inference."""
    try:
        import torch
        
        # Prefer Apple Silicon MPS on macOS
        if is_macos() and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        
        # Prefer CUDA if available
        if torch.cuda.is_available():
            return "cuda"
        
        return "cpu"
        
    except ImportError:
        return "cpu"


def get_system_info() -> dict:
    """Get comprehensive system information."""
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    
    # Add GPU information if available
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [torch.cuda.get_device_name(i) 
                               for i in range(torch.cuda.device_count())]
        
        if hasattr(torch.backends, 'mps'):
            info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        pass
    
    return info


def get_memory_info():
    """Get system memory information."""
    import psutil
    
    memory = psutil.virtual_memory()
    return {
        "total_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_gb": round(memory.used / (1024**3), 2),
        "percentage": memory.percent
    }


def is_admin() -> bool:
    """Check if the current user has administrator privileges."""
    try:
        if is_windows():
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        else:
            import os
            return os.geteuid() == 0
    except Exception:
        return False