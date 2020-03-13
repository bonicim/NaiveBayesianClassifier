from setuptools import find_packages, setup

setup(
    name="naive-bayesian-classifier",
    packages=["src"],
    version="0.1.0",
    url="https://github.com/bonicim/NaiveBayesianClassifier",
    author="Mark Bonicillo",
    author_email="markabonicillo@gmail.com",
    description="A homegrown Naive Bayes Classifier for classifying Tweets",
    long_description="",
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Data Science",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    entry_points={"console_scripts": ["nbc = src.nbc_cli:main"]},
)
