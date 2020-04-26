#!/bin/bash

set -e
set -u

# jupyter nbconvert --to python stage1_preprocessing_UMAFall.ipynb
# jupyter nbconvert --to python stage1_preprocessing_UPFall.ipynb
# jupyter nbconvert --to python stage1_preprocessing_SFDLA.ipynb
jupyter nbconvert --to python stage1_preprocessing_general.ipynb
