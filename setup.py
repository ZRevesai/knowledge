"""
Setup configuration for Knowledge-Guided Neural Networks (KGNN) package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kgnn-elderly-nutrition",
    version="0.1.0",
    author="Research Team",
    description="Knowledge-Guided Neural Networks for Micronutrient Deficiency Detection in Elderly Populations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZRevesai/knowledge",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=3.0.0",
            "black>=21.9b0",
            "flake8>=4.0.1",
            "mypy>=0.910",
        ],
    },
)
