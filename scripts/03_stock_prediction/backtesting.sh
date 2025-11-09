#!/bin/bash
# Define the list of train_months values
TRAIN_MONTHS_LIST=(12)

# Define the list of num_scenarios values
NUM_SCENARIOS_LIST=(300 25 50 75 100 150 500)

SECTOR_LIST=(InformationTechnology Materials ConsumerDiscretionary ConsumerStaples Energy Financials Healthcare Industrials RealEstate Utilities)  

# =========================================================
for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do
    for NUM_SCENARIOS in "${NUM_SCENARIOS_LIST[@]}"; do
        for SECTOR in "${SECTOR_LIST[@]}"; do

            echo "Running with SECTOR = ${SECTOR} | ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"        
            python run.py \
                --task_name Backtesting \
                --bt_name PlotResults \
                --model_name Transformer_13_64_3 \
                --exp_root_path SP500 \
                --sector ${SECTOR} \
                --test_start_year 2022 \
                --total_test_months 36 \
                --sliding_test_months 12 \
                --train_months ${TRAIN_MONTHS} \
                --num_scenarios ${NUM_SCENARIOS}
            
            echo "Finished running with SECTOR = ${SECTOR} | ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"
        done
    done
done