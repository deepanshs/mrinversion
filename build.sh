#!/bin/bash

source /Users/deepansh/opt/anaconda3/bin/activate

# python 3.6, 3.7
for PYBIN in py37; do
    conda create -n test python=3.6 -y
    conda activate test
    python --version
    pip install -r requirements.txt
    python setup.py develop
    pip install -r requirements-dev.txt
    pytest
done
