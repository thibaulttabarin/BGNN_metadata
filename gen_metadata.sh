#!/bin/bash

# generates metadata of enhanced model
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 gen_metadata.py /usr/local/bgnn/tulane
pipenv run python3 error_check.py

# generates metadata of non enhanced model
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 gen_metadata.py /usr/local/bgnn/tulane
pipenv run python3 error_check.py


# reset config
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
