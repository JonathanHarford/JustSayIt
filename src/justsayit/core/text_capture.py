"""
Text capture functionality for getting selected text from the system clipboard.
"""

import logging
import threading
import time
from typing import Optional, Callable, Set
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer, QObject, pyqtSignal


logger = logging.getLogger(__name__)


class TextCaptureMonitor(QObject):
    """Monitors clipboard for text selection changes."""
    
    # Signal emitted when new text is captured
    text_captured = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self._clipboard = QApplication.clipboard()
        self._last_text = ""
        self._monitoring = False
        self._monitor_timer = None
        
        # Callbacks for text capture events
        self._capture_callbacks: Set[Callable[[str], None]] = set()
        
        logger.debug("Text capture monitor initialized")
    
    def start_monitoring(self, interval_ms: int = 100):
        """Start monitoring clipboard for text changes."""
        if self._monitoring:
            return
        
        self._monitoring = True
        
        # Create timer for polling clipboard
        self._monitor_timer = QTimer()
        self._monitor_timer.timeout.connect(self._check_clipboard)
        self._monitor_timer.start(interval_ms)
        
        # Also connect to clipboard changed signal
        self._clipboard.dataChanged.connect(self._on_clipboard_changed)
        
        logger.info(f"Started clipboard monitoring (interval: {interval_ms}ms)")
    
    def stop_monitoring(self):
        """Stop monitoring clipboard."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        
        if self._monitor_timer:
            self._monitor_timer.stop()
            self._monitor_timer = None
        
        # Disconnect clipboard signal
        self._clipboard.dataChanged.disconnect(self._on_clipboard_changed)
        
        logger.info("Stopped clipboard monitoring")
    
    def get_current_selection(self) -> Optional[str]:
        """Get currently selected/clipboard text."""
        try:
            # Try to get text from clipboard
            mime_data = self._clipboard.mimeData()
            
            if mime_data.hasText():
                text = mime_data.text().strip()
                if text:
                    return text
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting clipboard text: {e}")
            return None
    
    def set_clipboard_text(self, text: str):
        """Set text to clipboard."""
        try:
            self._clipboard.setText(text)
            logger.debug(f"Set clipboard text: {text[:50]}...")
        except Exception as e:
            logger.error(f"Error setting clipboard text: {e}")
    
    def add_capture_callback(self, callback: Callable[[str], None]):
        """Add callback for text capture events."""
        self._capture_callbacks.add(callback)
        logger.debug(f"Added text capture callback (total: {len(self._capture_callbacks)})")
    
    def remove_capture_callback(self, callback: Callable[[str], None]):
        """Remove text capture callback."""
        self._capture_callbacks.discard(callback)
        logger.debug(f"Removed text capture callback (total: {len(self._capture_callbacks)})")
    
    def _check_clipboard(self):
        """Check clipboard for new text (polling method)."""
        try:
            current_text = self.get_current_selection()
            
            if current_text and current_text != self._last_text:
                self._handle_text_change(current_text)
                
        except Exception as e:
            logger.error(f"Error checking clipboard: {e}")
    
    def _on_clipboard_changed(self):
        """Handle clipboard changed signal."""
        try:
            current_text = self.get_current_selection()
            
            if current_text and current_text != self._last_text:
                self._handle_text_change(current_text)
                
        except Exception as e:
            logger.error(f"Error handling clipboard change: {e}")
    
    def _handle_text_change(self, text: str):
        """Handle detected text change."""
        # Filter out very short or common clipboard text
        if len(text) < 3:
            return
        
        # Skip if text looks like a single word (might be accidental)
        if len(text.split()) == 1 and len(text) < 20:
            return
        
        self._last_text = text
        
        logger.debug(f"New text captured: {text[:50]}...")
        
        # Emit signal
        self.text_captured.emit(text)
        
        # Call registered callbacks
        for callback in self._capture_callbacks:
            try:
                callback(text)
            except Exception as e:
                logger.error(f"Error in text capture callback: {e}")
    
    def is_monitoring(self) -> bool:
        """Check if currently monitoring."""
        return self._monitoring


class TextCapture:
    """High-level interface for text capture functionality."""
    
    def __init__(self):
        self._monitor = TextCaptureMonitor()
        self._auto_monitoring = False
    
    def get_selected_text(self) -> Optional[str]:
        """Get currently selected text immediately."""
        return self._monitor.get_current_selection()
    
    def start_auto_capture(self, callback: Callable[[str], None], interval_ms: int = 100):
        """Start automatically capturing text changes."""
        self._monitor.add_capture_callback(callback)
        
        if not self._auto_monitoring:
            self._monitor.start_monitoring(interval_ms)
            self._auto_monitoring = True
        
        logger.info("Started automatic text capture")
    
    def stop_auto_capture(self, callback: Optional[Callable[[str], None]] = None):
        """Stop automatic text capture."""
        if callback:
            self._monitor.remove_capture_callback(callback)
        
        # If no callbacks left, stop monitoring
        if not self._monitor._capture_callbacks and self._auto_monitoring:
            self._monitor.stop_monitoring()
            self._auto_monitoring = False
        
        logger.info("Stopped automatic text capture")
    
    def simulate_text_selection(self, text: str):
        """Simulate text selection for testing purposes."""
        logger.debug(f"Simulating text selection: {text[:50]}...")
        self._monitor._handle_text_change(text)
    
    def get_monitor(self) -> TextCaptureMonitor:
        """Get the underlying monitor object for advanced usage."""
        return self._monitor
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up text capture...")
        self._monitor.stop_monitoring()
        self._monitor._capture_callbacks.clear()