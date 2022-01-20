#!/bin/bash

# 10000 iterations
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
yq e -i '.SOLVER.MAX_ITER = 10000 | .SOLVER.STEPS = "(8200, 9000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py
./gen_metadata.sh

# 15000 iterations
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
yq e -i '.SOLVER.MAX_ITER = 15000 | .SOLVER.STEPS = "(12300, 13500)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py
./gen_metadata.sh


# 30000 iterations
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
yq e -i '.SOLVER.MAX_ITER = 30000 | .SOLVER.STEPS = "(24600, 27000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py
./gen_metadata.sh

# 50000 iterations
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
yq e -i '.SOLVER.MAX_ITER = 50000 | .SOLVER.STEPS = "(41000, 45000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py
./gen_metadata.sh

# 100000 iterations
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
yq e -i '.SOLVER.MAX_ITER = 50000 | .SOLVER.STEPS = "(82000, 90000)"' config/mask_rcnn_R_50_FPN_3x.yaml
pipenv run python3 train_model.py
jq '.ENHANCE = 0' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
pipenv run python3 train_model.py
./gen_metadata.sh

# reset
jq '.ENHANCE = 1' config/enhance.json > tmp.$$.json && mv tmp.$$.json config/enhance.json
