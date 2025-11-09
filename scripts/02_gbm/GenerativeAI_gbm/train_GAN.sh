#!/bin/bash

# Define the list of train_months values
MODEL_name=(VanillaGAN WGAN QuantGAN)

# SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials ) 
SECTOR_LIST=(Healthcare Industrials RealEstate InformationTechnology Utilities) 

#==============================================
for MODEL in "${MODEL_name[@]}"; do

    for SECTOR in "${SECTOR_LIST[@]}"; do    
        echo "Running with SECTOR = ${SECTOR} | MODEL_name = ${MODEL}"
        python run.py \
            --task_name train \
            --exp_root_path SP500 \
            --sector ${SECTOR} \
            --model_type GAN \
            --model_name "${MODEL}" \
            --test_start_year 2022 \
            --total_test_months 36 \
            --sliding_test_months 12 \
            --train_months 36 \
            --noise_input_size 3 \
            --noise_output_size 1 \
            --model_optimizer Adam \
            --learning_rate 0.0002 \
            --seq_len 127 \
            --batch_size 128 \
            --num_workers 0 \
            --num_epochs 5000 \
            --min_epochs 300 \
            --check_interval 10 \
            --fake_sample 10 \
            --loss_tolerance 0.0001 \
            --early_stop_patience 10 \
            --num_simulations 10 \
            --num_noise_samples 1

        echo "Finished running with SECTOR = ${SECTOR} | MODEL_name = ${MODEL}"
    
    done
done