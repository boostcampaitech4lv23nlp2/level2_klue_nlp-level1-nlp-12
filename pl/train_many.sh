#!/bin/bash
CONFIGS=("electra_focal05_origin_focal")

config_length=${#CONFIGS[@]}

for (( i=0; i<${config_length}; i++ ));
do
    echo ${CONFIGS[$i]}
    python3 main_n.py \
        --config ${CONFIGS[$i]}
done