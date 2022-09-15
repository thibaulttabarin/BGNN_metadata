#!/bin/bash

# 10000 iterations
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
yq e -i '.SOLVER.MAX_ITER = 10000 | .SOLVER.STEPS = "(8200, 9000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py
mkdir -p output/enhanced_15000
mkdir -p output/non_enhanced_15000
cp -R output/enhanced_10000/* output/enhanced_15000/
cp -R output/non_enhanced_10000/* output/non_enhanced_15000/

# 15000 iterations
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
yq e -i '.SOLVER.MAX_ITER = 15000 | .SOLVER.STEPS = "(12300, 13500)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py
mkdir -p output/enhanced_30000
mkdir -p output/non_enhanced_30000
cp -R output/enhanced_15000/* output/enhanced_30000/
cp -R output/non_enhanced_15000/* output/non_enhanced_30000/

# 30000 iterations
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
yq e -i '.SOLVER.MAX_ITER = 30000 | .SOLVER.STEPS = "(24600, 27000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py
mkdir -p output/enhanced_50000
mkdir -p output/non_enhanced_50000
cp -R output/enhanced_30000/* output/enhanced_50000/
cp -R output/non_enhanced_30000/* output/non_enhanced_50000/

# 50000 iterations
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
yq e -i '.SOLVER.MAX_ITER = 50000 | .SOLVER.STEPS = "(41000, 45000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py
mkdir -p output/enhanced_100000
mkdir -p output/non_enhanced_100000
cp -R output/enhanced_50000/* output/enhanced_100000/
cp -R output/non_enhanced_50000/* output/non_enhanced_100000/

# 100000 iterations
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
yq e -i '.SOLVER.MAX_ITER = 100000 | .SOLVER.STEPS = "(82000, 90000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
pipenv run python3 train_model.py

# reset
jq '.ENHANCE = 1' config/config.json > tmp.$$.json && mv tmp.$$.json config/config.json
