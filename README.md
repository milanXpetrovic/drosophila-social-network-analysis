# Drosophila social network analysis: 

**Python pipeline for *Drosophila melanogaster* social network analysis from trajectorial data**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10355543.svg)](https://doi.org/10.5281/zenodo.10355543)
[![License](https://img.shields.io/badge/license-BSD--3%20Clause-green)](https://github.com/milanXpetrovic/my_module/blob/main/LICENSE.md)

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626.svg?style=flat&logo=Jupyter)](https://jupyterlab.readthedocs.io/en/stable)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)


The module is being used on data obtained by monitoring Dosophila melanogaster populations and social interaction networks are being created from track data.

**Example of video recording:**

![](./docs/arena.gif)

## Complex networks features

- Static and snapshot network construction
- Static and dynamic analysis of the networks
- Global measures through time
- Comparison of measures between graphs from the same treatment
- Comparison of several treatments
- Centrality measures distribution

## Data processing features

- Functions for loading, validating, and preprocessing data.
- Functions for creating trajectory descriptors and ML features.
- Population analysis, distances and angles between individuals from the trajectory

## How to use?

1. Install needed packages:

```
pip install -r requirements.txt
```

2. Add tracking data to `.data/trackings/<treatment name>`

3. Check `configs/main.toml` for `START`, `END`, `ARENA_DIAMETER`, `FPS`.

4. Create configuration in `configs/trackings/<treatment_name>.toml` values for `angle`, `degree` and `time` which are the criteria for determining edges in networks.

5. Run `__main__.py` from `./src` all results will be saved to `./data/results`

## Join us
The project is open-source and aims to create research tools. If you are interested in development, feel free to contact us or just fork the repo.
