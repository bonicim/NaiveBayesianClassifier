from setuptools import find_packages, setup

setup(
    name="naive-bayesian-classifier",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="0.1.0",
)
