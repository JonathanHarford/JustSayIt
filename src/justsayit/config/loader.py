"""
Configuration loader with YAML support, validation, and hot-reload capabilities.
"""

import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Lock

from .schema import AppConfig, get_default_config_path, ensure_config_dir


logger = logging.getLogger(__name__)


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches config file for changes and triggers reload."""
    
    def __init__(self, config_loader: 'ConfigLoader'):
        self.config_loader = config_loader
        self.last_modified = 0
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
            
        if Path(event.src_path).name == "config.yaml":
            # Debounce rapid file changes
            import time
            current_time = time.time()
            if current_time - self.last_modified > 1.0:  # 1 second debounce
                self.last_modified = current_time
                logger.info("Configuration file changed, reloading...")
                self.config_loader.reload_config()


class ConfigLoader:
    """Handles loading, saving, and watching configuration files."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or get_default_config_path()
        self.config: Optional[AppConfig] = None
        self._config_lock = Lock()
        self._observer: Optional[Observer] = None
        self._callbacks = []
        
        # Ensure config directory exists
        ensure_config_dir()
    
    def load_config(self) -> AppConfig:
        """Load configuration from file or create default if not exists."""
        with self._config_lock:
            try:
                if self.config_path.exists():
                    logger.info(f"Loading configuration from {self.config_path}")
                    config_data = self._load_yaml_file(self.config_path)
                    self.config = AppConfig(**config_data)
                else:
                    logger.info("No configuration file found, creating default")
                    self.config = AppConfig()
                    self._save_default_config()
                    
                return self.config
                
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                logger.info("Using default configuration")
                self.config = AppConfig()
                return self.config
    
    def reload_config(self):
        """Reload configuration from file and notify callbacks."""
        try:
            old_config = self.config
            new_config = self.load_config()
            
            # Notify callbacks of config change
            for callback in self._callbacks:
                try:
                    callback(old_config, new_config)
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def save_config(self, config: AppConfig) -> bool:
        """Save configuration to file."""
        with self._config_lock:
            try:
                # Convert pydantic model to dict
                config_dict = config.dict()
                
                # Save to YAML file with comments
                self._save_yaml_file(self.config_path, config_dict)
                
                self.config = config
                logger.info(f"Configuration saved to {self.config_path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save configuration: {e}")
                return False
    
    def start_watching(self):
        """Start watching configuration file for changes."""
        if self._observer is not None:
            return  # Already watching
            
        try:
            self._observer = Observer()
            event_handler = ConfigFileWatcher(self)
            
            # Watch the config directory
            config_dir = self.config_path.parent
            self._observer.schedule(event_handler, str(config_dir), recursive=False)
            self._observer.start()
            
            logger.info(f"Started watching configuration file: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to start config file watching: {e}")
    
    def stop_watching(self):
        """Stop watching configuration file."""
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join()
                self._observer = None
                logger.info("Stopped watching configuration file")
            except Exception as e:
                logger.error(f"Error stopping config watcher: {e}")
    
    def add_change_callback(self, callback):
        """Add a callback to be called when config changes."""
        self._callbacks.append(callback)
    
    def remove_change_callback(self, callback):
        """Remove a config change callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_config(self) -> AppConfig:
        """Get current configuration (load if not loaded)."""
        if self.config is None:
            return self.load_config()
        return self.config
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML file and return as dictionary."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to read config file: {e}")
    
    def _save_yaml_file(self, file_path: Path, data: Dict[str, Any]):
        """Save dictionary as YAML file with nice formatting."""
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write header comment
                f.write("# JustSayIt Configuration File\\n")
                f.write("# This file is automatically generated and updated\\n")
                f.write("# You can edit it manually, but changes may be overwritten\\n\\n")
                
                # Write YAML data
                yaml.dump(
                    data,
                    f,
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False,
                    allow_unicode=True,
                    width=80
                )
        except Exception as e:
            raise RuntimeError(f"Failed to write config file: {e}")
    
    def _save_default_config(self):
        """Save default configuration to file."""
        try:
            # Copy from the bundled default config if available
            default_config_path = Path(__file__).parent.parent.parent.parent / "config" / "default_config.yaml"
            
            if default_config_path.exists():
                logger.info("Copying default configuration file")
                with open(default_config_path, 'r', encoding='utf-8') as src:
                    content = src.read()
                
                with open(self.config_path, 'w', encoding='utf-8') as dst:
                    dst.write(content)
            else:
                # Generate default config from schema
                logger.info("Generating default configuration from schema")
                self.save_config(AppConfig())
                
        except Exception as e:
            logger.warning(f"Failed to save default config: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_watching()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_watching()