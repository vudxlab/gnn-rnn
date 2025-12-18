"""Setup script for pca-aug-1dcnn-bigru package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pca-aug-1dcnn-bigru",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular deep learning framework for time series classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/pca-aug-1dcnn-bigru",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pca-prepare-data=scripts.prepare_data:main",
            "pca-train=scripts.train:main",
            "pca-evaluate=scripts.evaluate:main",
        ],
    },
)

