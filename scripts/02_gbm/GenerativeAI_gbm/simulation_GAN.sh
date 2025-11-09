#!/bin/bash
# Define the list of model names and train_months values
MODEL_name=(QuantGAN)
# MODEL_name=(VanillaGAN WGAN QuantGAN)

SECTOR_LIST=(InformationTechnology Utilities) 

# SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials Healthcare Industrials RealEstate InformationTechnology Utilities) 


# Loop over each train_months value
for MODEL in "${MODEL_name[@]}"; do

    for SECTOR in "${SECTOR_LIST[@]}"; do    
        echo "Running with SECTOR = ${SECTOR} | MODEL_name = ${MODEL}"
        python run.py \
            --task_name simulate \
            --exp_root_path SP500 \
            --sector ${SECTOR} \
            --model_type GAN \
            --model_name "${MODEL}" \
            --test_start_year 2022 \
            --total_test_months 36 \
            --sliding_test_months 12 \
            --train_months 12 \
            --noise_input_size 3 \
            --num_simulations 10 \
            --num_repeats 50 
        
         echo "Finished running with SECTOR = ${SECTOR} | MODEL_name = ${MODEL}"
    done
done
