#!/bin/bash

if [ $# != 1 ]; then
    echo "Invalid Usage. Please provide 1 integer argument between 0 and 2 inclusive."
    echo "0 represents aggregate, 1 represents INHS, 2 represents UWZM."
    exit 1
fi

case $1 in
    0) DIR="/usr/local/bgnn/tulane"; echo "Generating Aggregate Data";;
    1) DIR="/usr/local/bgnn/inhs_filtered"; echo "Generating Filtered INHS Data";;
    2) DIR="/usr/local/bgnn/uwzm_filtered"; echo "Generating Filtered UWZM Data";;
    *) echo "Invalid Usage. Please provide an integer between 0 and 2 inclusive."; exit 1;;
esac


# generates metadata of enhanced model
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 gen_metadata.py $DIR 

# generates metadata of non enhanced model
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 gen_metadata.py $DIR 

# reset config
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
./error_check.sh
