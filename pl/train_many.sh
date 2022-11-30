#!/bin/bash
CONFIGS=("roberta")

config_length=${#CONFIGS[@]}

for (( i=0; i<${config_length}; i++ ));
do
    echo ${CONFIGS[$i]}
    python3 main_n.py \
        --config ${CONFIGS[$i]}
done