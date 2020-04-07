#!/bin/bash

set -e
set -u

jupyter nbconvert --to python stage1_preprocessing_UMAFall.ipynb
