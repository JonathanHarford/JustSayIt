"""
JustSayIt - Cross-platform accessibility tool with offline TTS and OCR
"""
import os
from setuptools import setup, find_packages

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="justsayit",
    version="0.1.0",
    description="Cross-platform accessibility tool with offline TTS and OCR capabilities",
    long_description=read_readme() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    author="JustSayIt Team",
    author_email="contact@justsayit.dev",
    url="https://github.com/justsayit/justsayit",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt") if os.path.exists("requirements.txt") else [],
    extras_require={
        "dev": read_requirements("requirements-dev.txt") if os.path.exists("requirements-dev.txt") else [],
    },
    entry_points={
        "console_scripts": [
            "justsayit=justsayit.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    keywords="accessibility tts text-to-speech ocr screen-reader offline",
    project_urls={
        "Bug Reports": "https://github.com/justsayit/justsayit/issues",
        "Source": "https://github.com/justsayit/justsayit",
        "Documentation": "https://justsayit.readthedocs.io/",
    },
    include_package_data=True,
    package_data={
        "justsayit": [
            "config/*.yaml",
            "assets/*",
            "assets/icons/*",
        ],
    },
)