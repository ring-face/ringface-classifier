
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='classifier-refit',
    version='0.1.0',
    description='',
    long_description=readme,
    author='Csaba Meszaros',
    author_email='abax.soraszem@gmail.com',
    url='https://github.com/xxx',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

