#!/bin/bash

# train with contrast config
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py

# train without contrast config
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py

# reset config and gen metadata
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
./gen_metadata.sh 1
./gen_metadata.sh 2
