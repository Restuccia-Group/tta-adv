#!/bin/bash

# Define the arrays for each argument
attacks=("fca")
tta_algorithms=("tent" "eata" "sotta" "sar")
severities=(1 2 3 4 5)

# Loop through each combination of arguments
for attack in "${attacks[@]}"; do
  for tta_algorithm in "${tta_algorithms[@]}"; do
    for severity in "${severities[@]}"; do
      # Run the Python script with the current combination of arguments
      python3 main.py --attack "$attack" --tta "$tta_algorithm" --severity "$severity" --batch_size 200 --gpu_id 1 --dataset cifar100c
    done
  done
done
