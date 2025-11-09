#!/bin/bash
# Define the list of train_months values
TRAIN_MONTHS_LIST=(12)

# Define the list of num_scenarios values
# NUM_SCENARIOS_LIST=($(seq 25 25 100))

NUM_SCENARIOS_LIST=(150 300 500)

# GBM_type_LIST=(Mathematical/standard_gbm Mathematical/empirical_gbm GAN/VanillaGAN)
GBM_type_LIST=(GAN/WGAN GAN/QuantGAN Diffusion/DDPM)

# =========================================================
for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do
    for GBM_type in "${GBM_type_LIST[@]}"; do
        for NUM_SCENARIOS in "${NUM_SCENARIOS_LIST[@]}"; do

            echo "Running with GBM_type = ${GBM_type} | ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"        
            python run.py \
                --task_name stock_prediction \
                --GBM_type ${GBM_type} \
                --exp_root_path SP500 \
                --sector Utilities \
                --test_start_year 2022 \
                --total_test_months 36 \
                --sliding_test_months 12 \
                --train_months ${TRAIN_MONTHS} \
                --seq_len 127 \
                --batch_size 128 \
                --num_epochs 10 \
                --num_scenarios ${NUM_SCENARIOS}
            
            echo "Finished running with GBM_type = ${GBM_type} | ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"
        done
    done
done

# SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials) 
# SECTOR_LIST=(Healthcare Industrials RealEstate InformationTechnology Utilities) 

# # =========================================================
# for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do
#     for SECTOR in "${SECTOR_LIST[@]}"; do
#         for NUM_SCENARIOS in "${NUM_SCENARIOS_LIST[@]}"; do

#             echo "Running with SECTOR = ${SECTOR} | ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"        
#             python run.py \
#                 --task_name stock_prediction \
#                 --GBM_type Diffusion/DDPM \
#                 --exp_root_path SP500 \
#                 --sector ${SECTOR} \
#                 --test_start_year 2022 \
#                 --total_test_months 36 \
#                 --sliding_test_months 12 \
#                 --train_months ${TRAIN_MONTHS} \
#                 --seq_len 127 \
#                 --batch_size 128 \
#                 --num_epochs 10 \
#                 --num_scenarios ${NUM_SCENARIOS}
            
#             echo "Finished running with SECTOR = ${SECTOR} | ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"
#         done
#     done
# done
