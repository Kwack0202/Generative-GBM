#!/bin/bash
# Define the list of train_months values
TRAIN_MONTHS_LIST=(12)

# Define the list of num_scenarios values
SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials Healthcare Industrials RealEstate InformationTechnology Utilities) 

# =========================================================
for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do
    for SECTOR in "${SECTOR_LIST[@]}"; do

        echo "Running with SECTOR = ${SECTOR} | ${TRAIN_MONTHS}"        
        python run.py \
            --task_name baseline_TF \
            --exp_root_path SP500 \
            --sector ${SECTOR} \
            --test_start_year 2022 \
            --total_test_months 36 \
            --sliding_test_months 12 \
            --train_months ${TRAIN_MONTHS} \
            --seq_len 127 \
            --batch_size 1024 \
            --num_epochs 10
        
        echo "Finished running with SECTOR = ${SECTOR} | ${TRAIN_MONTHS}"
        
    done
done