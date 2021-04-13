rom setuptools import setup, find_packages
# To use a consistent encoding
# from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
# with open('README.rst') as f:
#     long_description = f.read()

setup(
    name='pygorithm',
    version='0.0.420',
    description='Toolkit for trajectory data processing, network construction and analysis ',
    long_description=open('README.md').read(),
    # The project's main homepage.
    url='https://github.com/milky-boi/my_module',
    # Author details
    author='milky-boi',
    author_email='',
    # Choose your license
    #! FIX THIS
    license='MIT',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: ',
        'Topic :: Data processing :: Complex networks :: Libraries',

        # Pick your license as you wish (should match "license" above)
        #! FIX THIS
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.

        'Programming Language :: Python :: 3.8',
    ],
    packages=find_packages()
)