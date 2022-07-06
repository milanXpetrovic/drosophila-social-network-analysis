# My module: Toolkit for trajectory data processing,network construction and analysis

[![License](https://img.shields.io/badge/license-BSD--3%20Clause-green)](https://github.com/milanXpetrovic/my_module/blob/main/LICENSE.md)


"""
Python module
Purpose of this Python module is to create toolkit for analysing data from
complex systems. Analysis is from the data collected by monitoring the participants
in that system, i.e. individuals. 

The project is open-source and aims to create research tools.
If you are interested in development, feel free to contact us.

Currently the module is being tested on data created by monitoring the biological
system (drosophila melanogaster populations) and social interaction networks are being created.

LIST OF SUB-MODULES:

- Toolkit: Functions for data manipulation, organizing and reading the
contents of a large number of folders and files within them or the files themselves.
Another option is to check the validity of the data, or the missing records, 
then the possibility of cleaning or fixing them.

- Ml: functionalities for data transformation (eg extracting statistical values,
smoothing data, etc.) and creating forms that for process and transform data 
for implementation of machine learning.

- Networks: Analysis of individuals within a biologically complex systems.
Interpretation of data in the form of complex networks and the presentation
of data through graphs will be enabled. And the second part of the sub-module
contains functions for analysis and calculations over the created graphs.
In the last part of this module, there would be machine learning functions on graphs.

"""

*Components of this module are:*
- Data loading, validity checking and preprocessing functions
- Path descriptors and ML features creating functions
- Population analysis, distances between elements from trajectory
- Complex network construction
- Complex network analysis
