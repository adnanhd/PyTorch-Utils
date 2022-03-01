from setuptools import setup
from __about__ import __author__, __email__

with open('README.md') as f:
    long_description = f.readlines()

with open('requirements.txt') as f:
    required_packages = f.readlines()

setup(
    name='torchutils',
    version='1.0',
    description=long_description,
    author=__author__,
    author_email=__email__,
    install_requires=required_packages,
)

