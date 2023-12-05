# Creature Generator

This project aimed at training a neural network to predict Base64 files based on natural language descriptions. Leveraging the power of machine learning and neural networks, this project explores the correlation between textual descriptions and the corresponding encoded Base64 files.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Features

This project has two main folders

- dataset_creation: has the files to scrap the data and create the dataset that can be found [here](https://brpucrs-my.sharepoint.com/:t:/g/personal/guido_mainardi_edu_pucrs_br/EZ3IMvw82blKpiEyi9Wn6_wB2ZD_22vLo6_5K4QThV2z8A?e=UhIW5z).
- custom_nn: has the attention model files and data 

## Installation

First of all you need to make sure you have python 3.9 with pip installed.


```bash
# Example installation steps or commands
$ pip install poetry
$ cd creature-generator/custom_nn
$ poetry install
```

The dataset creation dependencies are not menage by poetry, so you might have to install some packages in order to run these scripts.

## Usage

Once you have installed all the dependencies, running the code is realy simple. 

1. open the custom_nn/training.ipynb on your IDE
2. run the cells to start the training process 
