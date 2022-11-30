#!/bin/bash
CONFIGS=("electra_ce_origin" "electra_label_origin" "electra_ce_yong" "electra_label_yong")

config_length=${#CONFIGS[@]}

for (( i=0; i<${config_length}; i++ ));
do
    echo ${CONFIGS[$i]}
    python3 main_n.py \
        --config ${CONFIGS[$i]}
done