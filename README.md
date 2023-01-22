# ARTEMIS: A Python Library for Feature Interactions in Machine Learning Models
[![build](https://github.com/pyartemis/artemis/actions/workflows/python-app.yml/badge.svg)](https://github.com/pyartemis/artemis/actions/workflows/python-app.yml)
[![PyPI version](https://badge.fury.io/py/pyartemis.svg)](https://pypi.org/project/pyartemis/)
[![Downloads](https://static.pepy.tech/badge/pyartemis)](https://pepy.tech/project/pyartemis)

## Overview
`artemis` is a **Python** package for data scientists and machine learning practitioners which exposes standardized API for extracting feature interactions from predictive models using a number of different methods described in scientific literature.

The package provides both model-agnostic (no assumption about model structure), and model-specific (e.g., tree-based models) feature interaction methods, as well as other methods that can facilitate and support the analysis and exploration of the predictive model in the context of feature interactions. 

The available methods are suited to tabular data and classification and regression problems. The main functionality is that users are able to scrutinize a wide range of models by examining feature interactions in them by finding the strongest ones (in terms of numerical values of implemented methods) and creating tailored visualizations.

## Documentation
Full documentation is available at [https://pyartemis.github.io/](https://pyartemis.github.io/).

## Installation
Latest released version of the `artemis` package is available on [Python Package Index (PyPI)](https://pypi.org/project/pyartemis/):

```
pip install -U pyartemis
```

The source code and development version is currently hosted on [GitHub](https://github.com/pyartemis/artemis).

***

## Authors

The package was created as a software project associated with the BSc thesis ***Methods for extraction of interactions from predictive models*** in the field of Data Science (pl. *Inżynieria i analiza danych*) at Faculty of Mathematics and Information Science (MiNI), Warsaw University of Technology. 

The authors of the `artemis` package are: 
- [Paweł Fijałkowski](https://github.com/pablo2811)
- [Mateusz Krzyziński](https://github.com/krzyzinskim)
- [Artur Żółkowski](https://github.com/arturzolkowski)

BSc thesis and work on the `artemis` package was supervised by [Przemysław Biecek, PhD, DSc](https://github.com/pbiecek). 

