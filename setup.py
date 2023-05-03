from os import path
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="drosophila-social-network-analysis",
    version="0.0",
    description="Toolkit for trajectory Fruit fly data processing.",
    long_description=long_description,
    url="https://github.com/milanXpetrovic/drosophila-social-network-analysis",
    author="milanXpetrovic",
    author_email="milan.petrovic@uniri.hr",
    license="MIT",
    classifiers=[
        "Topic :: Drosophila Melanogaster",
    ],
    keywords=["trajectory", "data-analysis"],
    package_dir={"src": "src"},
    packages=["src", "src.utils"],
    install_requires=["pandas>1.5"],
    project_urls=dict(Documentation="", Source="", Issues="", Changelog=""),
)
