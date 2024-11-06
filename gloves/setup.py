from setuptools import setup, find_packages

setup(
    name='gloves',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'yfinance',
        'pandas',
    ],
    author='Bruce McNair',
    description='Utility package. Very basic. First thing you do in a lab is put on gloves.',
)
