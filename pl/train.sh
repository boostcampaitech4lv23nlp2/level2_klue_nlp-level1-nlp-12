#!/bin/bash
CONFIGS=("rroberta")

config_length=${#CONFIGS[@]}

fold="fold"
basic="basic"

for (( i=0; i<${config_length}; i++ ));
do
    if [ $1 = $fold ]
    then
        echo ${CONFIGS[$i]}
        python3 r_main.py \
            --config ${CONFIGS[$i]}
    elif [ $1 = $basic ]
    then
        echo ${CONFIGS[$i]}
        python3 r_main.py \
            --config ${CONFIGS[$i]}
    fi
done