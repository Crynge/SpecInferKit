from setuptools import setup, find_packages

setup(
    name="specinferkit",
    version="0.1.0",
    description="Production-grade speculative decoding toolkit for LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Sameer Alam",
    url="https://github.com/Crynge/SpecInferKit",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "ruff>=0.1.0", "mypy>=1.7.0"],
        "serving": ["fastapi>=0.104.0", "uvicorn>=0.24.0"],
        "all": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "pytest>=7.4.0"],
    },
    entry_points={
        "console_scripts": [
            "specinferkit=specinferkit.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
