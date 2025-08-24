"""
JustSayIt main application entry point.
"""

import sys
import logging
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from justsayit.ui.system_tray import SystemTrayApp
from justsayit.config.loader import ConfigLoader


def setup_logging():
    """Set up application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Path.home() / '.justsayit' / 'justsayit.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)  # Keep running in system tray
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.load_config()
    
    # Create and run the system tray application
    tray_app = SystemTrayApp(config)
    
    if not tray_app.is_system_tray_available():
        logger.error("System tray not available on this platform")
        sys.exit(1)
    
    tray_app.show()
    logger.info("JustSayIt started successfully")
    
    # Use a timer to process events periodically
    timer = QTimer()
    timer.timeout.connect(lambda: None)  # Keep the event loop active
    timer.start(100)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()