"""
Setup configuration for bug-severity-classification package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
dev_requirements = (this_directory / "requirements-dev.txt").read_text().splitlines()

setup(
    name="bug-severity-classification",
    version="1.0.0",
    author="Niisa",
    author_email="your.email@example.com",
    description="ML-powered bug severity classification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bug-severity-classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "bug-predict=scripts.predict:main",
        ],
    },
)