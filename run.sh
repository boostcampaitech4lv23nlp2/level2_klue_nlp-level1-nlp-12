#!/bin/bash

while  read line
do 
    python train.py $line
    python inference.py $line
    
done < command_file.txt