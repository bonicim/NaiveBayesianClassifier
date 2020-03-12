from setuptools import find_packages, setup

setup(
    # required
    name="naive-bayesian-classifier",
    packages=["src"],
    # package_dir={"": "src"},
    version="0.1.0",
    url="https://github.com/bonicim/NaiveBayesianClassifier",
    # Metadata
    author="Mark Bonicillo",
    author_email="markabonicillo@gmail.com",
    description="A simple implementation of a Naive Bayes Classifier",
    long_description="",
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Data Science",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points={"console_scripts": ["nbc = src.nbc_cli:main"]},
)
