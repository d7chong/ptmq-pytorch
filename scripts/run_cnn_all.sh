#!/bin/bash

models=(resnet18 resnet50 regnetx_600m mobilenetv2)
declare -A config
config=( 
    [3]="al=3 am=6 ah=8"
    [4]="al=4 am=6 ah=8"
    [6]="al=6 am=7 ah=8"
    [8]="al=6 am=7 ah=8"
)

learning_rate=4e-3
iterations=5000

for model in "${models[@]}"; do
    for w in "${!config[@]}"; do
        al=$(echo ${config[$w]} | awk '{print $1}' | cut -d'=' -f2)
        am=$(echo ${config[$w]} | awk '{print $2}' | cut -d'=' -f2)
        ah=$(echo ${config[$w]} | awk '{print $3}' | cut -d'=' -f2)
        python run_ptmq.py -m "$model" -w "$w" -a "$w" -al "$al" -am "$am" -ah "$ah" -lr "$learning_rate" -i "$iterations"
    done
done