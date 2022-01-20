#!/bin/bash

# tests enhanced data
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 error_check.py

# tests non enhanced data
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 error_check.py

# reset config
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
