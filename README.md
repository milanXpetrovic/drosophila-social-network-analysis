# My module: Toolkit for trajectory data processing,network construction and analysis

[![License](https://img.shields.io/badge/license-BSD--3%20Clause-green)](https://github.com/milanXpetrovic/my_module/blob/main/LICENSE.md)


The purpose of this Python module is to create a toolkit for analysing tracking data. The analysis is done using the data collected by monitoring the participants in given system, i.e. individuals. 

The project is open-source and aims to create research tools. If you are interested in development, feel free to contact us.

Currently, the module is being tested on data obtained by monitoring Dosophila melanogaster populations and social interaction networks are being created from track data.

## Module components
- Functions for loading, validating, and preprocessing data.
- Functions for creating trajectory descriptors and ML features.
- Population analysis, distances between elements from the trajectory
- Complex network construction
- Complex network analysis

## List of submodules

- Toolkit: functions for data manipulation, organising and reading the contents of a large number of folders and files in them or the files themselves.
Another option is to check the validity of the data or the missing records and the possibility to clean or correct them.

- ML: functions for data transformation (e.g. extracting statistical values, smoothing data, etc.) and creating forms for processing and transforming data for implementing machine learning.

- Networks: analysing individuals within a biologically complex system.
Interpreting data in terms of complex networks and representing data through graphs is enabled. And the second part of the submodule contains functions for analysis and calculations over the created graphs.

The last part of this module provides functions for machine learning on graphs (currently being developed).

## How to use it

Just import 'my_module.py' from './src' and create your own data processing pipeline from its functions.

## Data

Currently, it is possible to test the functionality of the module using tracking data from Drosophila melanogaster. All data are freely available for further analysis and testing.

Each folder is a population, the folder name is structured as `<treatment or baseline>_<date>_<time>`. Within each folder (population) are tracking files of individuals. They are in .csv format and contain the x and y position coordinates for each image for each individual frame from tracked video.
