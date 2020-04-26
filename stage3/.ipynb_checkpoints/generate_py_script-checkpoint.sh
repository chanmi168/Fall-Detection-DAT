#!/bin/bash

set -e
set -u

jupyter nbconvert --to python stage3_model_eval.ipynb

