# Drosophila social network analysis: 

**Python pipeline for *Drosophila melanogaster* social network analysis from trajectorial data**

[![License](https://img.shields.io/badge/license-BSD--3%20Clause-green)](https://github.com/milanXpetrovic/my_module/blob/main/LICENSE.md)

Currently, the module is being tested on data obtained by monitoring Dosophila melanogaster populations and social interaction networks are being created from track data.

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
