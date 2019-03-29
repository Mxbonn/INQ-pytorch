import os
import io
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


readme = read('README.md')

setup(
    name='inq',
    packages=find_packages(),
    version='0.1.0',
    description='Incremental Network Quantization library for PyTorch.',
    long_description=readme,
    author='Maxim Bonnaerens',
    author_email='maxim@bonnaerens.be',
    requirements=['pytorch'],
)
