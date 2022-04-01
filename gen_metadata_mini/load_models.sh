#!/usr/bin/bash

mkdir output
mkdir output/enhanced
mkdir output/non_enhanced

pip install gdown -y
gdown -O output/enhanced/ https://drive.google.com/uc?id=13pa5E5odN_gWNZYkA12u8ZEnEjzWGxFL
mv output/enhanced/* output/enhanced/model_final.pth

gdown -O output/non_enhanced/ https://drive.google.com/uc?id=1YHGUkY1EUAfeHHd3crZToQMIrseW6INo
mv output/non_enhanced/* output/non_enhanced/model_final.pth

gdown -O output/ https://drive.google.com/uc?id=1QWzmHdF1L_3hbjM85nOjfdHsm-iqQptG
mv output/* output/model_final.pth
