#!/bin/bash

set -e
set -u

# jupyter nbconvert --to python stage1_preprocessing_general.ipynb
jupyter nbconvert --to python stage1_preprocessing_general_aug.ipynb
