"""
Setup configuration for NUTRINET package.

NUTRINET: A Computationally Efficient Graph Neural Model
for Interpretable Nutrient Interaction Analysis
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "NUTRINET: Efficient and interpretable nutrient interaction analysis"

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ]

setup(
    name="nutrinet",
    version="1.0.0",
    author="Zvinodashe Revesai and Okuthe P. Kogeda",
    author_email="224195689@stu.ukzn.ac.za",
    description="A Computationally Efficient Graph Neural Model for Interpretable Nutrient Interaction Analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZRevesai/nutrinet",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nutrinet-train=train:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "graph neural networks",
        "nutritional informatics",
        "explainable AI",
        "computational efficiency",
        "green AI",
        "vulnerable populations",
        "nutrient interactions",
        "deficiency detection",
    ],
)
