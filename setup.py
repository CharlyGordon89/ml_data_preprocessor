# setup.py

from setuptools import setup, find_packages

setup(
    name="ml_preprocessor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "scikit-learn>=1.0.0"
    ],
    author="Ruslan Mamedov",
    description="Reusable data preprocessing module for ML pipelines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    python_requires=">=3.7"
)
