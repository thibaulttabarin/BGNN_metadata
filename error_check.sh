#!/bin/bash

# tests enhanced data
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 error_check.py

# tests non enhanced data
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 error_check.py

# reset config
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
