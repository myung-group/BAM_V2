from setuptools import find_packages, setup

setup(
    name="BAM-nequip",
    version="0.1.0.dev0",
    author="Myung Group",
    author_email="cwmyung@skku.edu",
    description="Uncertainty Quantification of Bayesian Deep Learning ab initio Potential",
    url="https://github.com",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT ",
    ],
    install_requires=[
        "optax",
        "flax",
        "ase",
        "matscipy",
        "jraph",
        "e3nn-jax"
    ],
)



