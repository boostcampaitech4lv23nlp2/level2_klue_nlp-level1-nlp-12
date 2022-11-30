#!/bin/bash
<<<<<<< HEAD
CONFIGS=("hanja_config")
=======
CONFIGS=("roberta")
>>>>>>> v1.4.4

config_length=${#CONFIGS[@]}

fold="fold"
basic="basic"

for (( i=0; i<${config_length}; i++ ));
do
    if [ $1 = $fold ]
    then
        echo ${CONFIGS[$i]}
        python3 main.py \
            --config ${CONFIGS[$i]}
    elif [ $1 = $basic ]
    then
        echo ${CONFIGS[$i]}
        python3 main_n.py \
            --config ${CONFIGS[$i]}
    fi
done