#!/bin/bash

config_names="lgbm xgb cb"
is_tuned="tuned"
num_trials=100
modes="tune train test predict"

for mode in $modes
do
    for config_name in $config_names
    do
        python main.py --config-name=$config_name.yaml \
            mode=$mode \
            is_tuned=$is_tuned \
            num_trials=$num_trials
    done
done
