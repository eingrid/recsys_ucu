#!/bin/bash

# Download the dataset
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip 

# Unzip the dataset
unzip ml-1m.zip -d .

# Remove the zip file
rm ml-1m.zip

