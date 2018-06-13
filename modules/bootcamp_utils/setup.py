import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='bootcamp_utils',
    version='0.0.1',
    author='Justin Bois',
    author_email='bois@caltech.edu',
    description='Utilities for use in bootcamp.',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)
