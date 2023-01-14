# An Automated Prior Robustness Analysis in Bayesian Model Comparison

Project for the third year class "Bayesian Statistics" at ENSAE IP Paris.

* Benjamin Maurel
* Gabriel Watkinson

## Introduction

In this project, we implement the methods described in the paper [An Automated Prior Robustness Analysis in Bayesian Model Comparison](https://joshuachan.org/papers/AD_ML.pdf) by Joshua Chan, Liana Jacobi and Dan Zhu, published in the Journal of Applied Econometrics in 2022.

A short report is associated with this repository, and can be found [here](), describing the methods and the results obtained.

## Files and folders

* The implementation is done in Python, and the code is available in the `src` folder.
* The notebooks used to generate the results are available in the `notebooks` folder.

## The data used

The data used in the project is available in the `data` folder. The dataset `USdata_2019Q4.xlsx` is taken from the **Federal Reserve Bank of St. Louis' FRED-QD database** and the sample period is **1948:Q1-2019:Q4**. The first column of the file contains the dates. The second to last columns contain the values of GDP deflator, unemployment rate, real GDP growth and Fed funds rate, respectively.

## The report and pdfs

The [report]() is available in the `pdfs` folder, as well as the original paper and the instructions of the project.

For more ressources, please visit the author's website: http://joshuachan.org/research.html. You can directly download their Matlab code and the datasets used here : http://joshuachan.org/code/AD_ML_code.zip. The paper is also available : http://joshuachan.org/papers/AD_ML.pdf.

## Running the code

If you want to rerun the experiments, you can clone the project and run the notebooks in the `notebooks` folder.

```bash
git clone https://github.com/gwatkinson/automated_prior_analysis.git
```

### Reproducing the environment

The environment used to run the code is available in the `poetry.lock` file, it is generated with the `pyproject.toml` file.

If you have [poetry installed](https://python-poetry.org/docs/), you can create the environment using the following command:

```bash
poetry install
```

This will create a virtual environment with all the dependencies taht are specified in the lock file, needed to run the code. It will also install the directory as a package, so that you can import the modules in the `src` folder.

Alternatively, you can install the dependencies manually using `pip`:

```bash
pip install -r requirements.txt
```
This will install the dependencies in your current environment.

### Pre-commit

If you want to contribute, please install the pre-commit hooks (in the root folder with git and with the enviromnent activated):

```
pre-commit install
```

This installs hooks to /.git/hooks and run it once against the code:

```bash
pre-commit run --all-files
```

This will run some formatters and other hooks before each commit.
