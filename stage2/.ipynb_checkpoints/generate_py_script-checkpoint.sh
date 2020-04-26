#!/bin/bash

set -e
set -u

# jupyter nbconvert --to python stage2_DANN_HPsearch.ipynb
# jupyter nbconvert --to python stage2_DANN_rep10.ipynb
jupyter nbconvert --to python stage2_DANN_HPsearch_rep.ipynb
