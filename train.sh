#!/bin/bash

# train with contrast enhance
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py

# train without contrast enhance
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py

# reset config and gen metadata
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
./gen_metadata.sh
