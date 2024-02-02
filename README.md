# Deconvoluting Low Yield from Weak Potency in Direct-to-Biology Workflows with Machine Learning

This repo contains the data and code needed to reproduce the methodology and results for the paper "Deconvoluting Low Yield from Weak Potency in Direct-to-Biology Workflows with Machine Learning".

The necessary packages needed can be install by running `pip install .` in the directory of the repo.

The folder `data` contains the raw experimental data from both the initial direct-to-biology screen as well as the subsequent prospective screens.

The folder `scripts` contains the python scripts used for training and inferencing the machine learning models used.

The folder `notebooks` contains notebooks for generating the figures and calculating the metrics in the paper, as well as the cheminformatics code used to enumerate building blocks from EnamineREAL.