#!/bin/bash
conda env export > environment.yml --no-build &&
pip list --format=freeze > requirements.txt