#!/bin/bash

# Define the list of train_months values
TRAIN_MONTHS_LIST=(12)
# TRAIN_MONTHS_LIST=(12 24 36)

SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials Healthcare Industrials RealEstate InformationTechnology Utilities) 
# SECTOR_LIST=(Healthcare Industrials RealEstate InformationTechnology Utilities) 


#==============================================
for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do

    for SECTOR in "${SECTOR_LIST[@]}"; do    
        echo "Running with SECTOR = ${SECTOR} | train_months = ${TRAIN_MONTHS}"
        python run.py \
            --task_name train \
            --exp_root_path SP500 \
            --sector ${SECTOR} \
            --model_type Diffusion \
            --model_name LDM \
            --test_start_year 2022 \
            --total_test_months 36 \
            --sliding_test_months 12 \
            --train_months ${TRAIN_MONTHS} \
            --noise_input_size 3 \
            --noise_output_size 1 \
            --model_optimizer Adam \
            --learning_rate 0.0002 \
            --seq_len 127 \
            --batch_size 128 \
            --num_workers 0 \
            --num_epochs 5000 \
            --min_epochs 1 \
            --check_interval 10 \
            --fake_sample 10 \
            --loss_tolerance 0.0001 \
            --early_stop_patience 20 \
            --num_simulations 10 \
            --num_noise_samples 1 \
            --beta_start 0.0001 \
            --beta_end 0.02 \
            --T_diffusion 200
            
        echo "Finished running with SECTOR = ${SECTOR} | train_months = ${TRAIN_MONTHS}"
    
    done
done
