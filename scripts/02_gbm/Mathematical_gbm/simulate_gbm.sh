#!/bin/bash
# Define the list of train_months values
GBM_name=(standard_gbm)
# GBM_name=(standard_gbm empirical_gbm)

TRAIN_MONTHS_LIST=(12)
SECTOR_LIST=(InformationTechnology) 
# SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials Healthcare Industrials RealEstate InformationTechnology Utilities) 

#==============================================
for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do

    for GBM in "${GBM_name[@]}"; do

        for SECTOR in "${SECTOR_LIST[@]}"; do    
            echo "Running with SECTOR = ${SECTOR} | GBM_name = ${GBM} | train_months = ${TRAIN_MONTHS}"
            python run.py \
                --task_name Mathematical_GBM \
                --exp_root_path SP500 \
                --sector ${SECTOR} \
                --gbm_name "${GBM}" \
                --test_start_year 2022 \
                --total_test_months 36 \
                --sliding_test_months 12 \
                --train_months ${TRAIN_MONTHS} \
                --num_simulations 50 \
                --num_repeats 1 

            echo "Finished running with SECTOR = ${SECTOR} | GBM_name = ${GBM} | train_months = ${TRAIN_MONTHS}"
        
        done
    done
done