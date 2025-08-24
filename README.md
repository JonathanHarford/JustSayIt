# JustSayIt

A cross-platform, hotkey-driven accessibility tool that provides text-to-speech (TTS) and on-screen optical character recognition (OCR) capabilities. JustSayIt operates entirely offline, prioritizing user privacy and performance by leveraging local, open-weight models.

## Features

- **Text-to-Speech**: Press a hotkey to have any selected text read aloud using local TTS models
- **Screen OCR**: Select any area of your screen to extract and read text from images, PDFs, or inaccessible interfaces  
- **Offline First**: All processing happens locally using open-weight models - no internet required
- **Cross-Platform**: Works on macOS, Linux, and Windows
- **Privacy Focused**: Your data never leaves your device
- **Customizable**: Configure hotkeys, voices, and settings via simple YAML files

## Quick Start

### Installation

```bash
# Install from PyPI (coming soon)
pip install justsayit

# Or install from source
git clone https://github.com/justsayit/justsayit.git
cd justsayit
pip install -e .
```

### Usage

1. **Start JustSayIt**: Run `justsayit` from the command line or find it in your system tray
2. **Read Selected Text**: Select any text and press `Ctrl+Shift+S`
3. **OCR Screen Area**: Press `Ctrl+Shift+A`, then click and drag to select an area
4. **Stop Speech**: Press `Ctrl+Shift+X` to stop current playback

### Configuration

Settings can be customized by editing the configuration file:

- **Linux/macOS**: `~/.justsayit/config.yaml`
- **Windows**: `%USERPROFILE%\.justsayit\config.yaml`

Or right-click the system tray icon and select "Edit Configuration".

## Development

### Setup Development Environment

```bash
git clone https://github.com/justsayit/justsayit.git
cd justsayit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/justsayit --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

### Project Structure

```
justsayit/
├── src/justsayit/           # Main application code
│   ├── core/               # TTS, OCR, and audio engines
│   ├── ui/                 # System tray and settings UI
│   ├── hotkeys/            # Global hotkey management
│   ├── config/             # Configuration handling
│   └── utils/              # Utilities and platform code
├── tests/                  # Test suite
├── config/                 # Default configuration
├── docs/                   # Documentation
└── assets/                 # Icons and resources
```

## Implementation Status

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed development progress and roadmap.

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built with [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for cross-platform GUI
- Uses [Hugging Face Transformers](https://huggingface.co/transformers/) for ML models
- TTS powered by [Coqui TTS](https://github.com/coqui-ai/TTS)
- OCR powered by [TrOCR](https://huggingface.co/microsoft/trocr-base-printed)

---

**Note**: JustSayIt is currently in active development. Some features may not be fully implemented yet. See the implementation plan for current status.