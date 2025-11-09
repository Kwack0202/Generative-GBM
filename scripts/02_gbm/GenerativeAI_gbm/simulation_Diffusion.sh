#!/bin/bash
# Define the list of model names and train_months values
MODEL_name=(LDM)
# MODEL_name=(DDPM LDM)

SECTOR_LIST=(InformationTechnology) 


# SECTOR_LIST=(Materials ConsumerDiscretionary ConsumerStaples Energy Financials Healthcare Industrials RealEstate InformationTechnology Utilities) 

# Loop over each train_months value
for MODEL in "${MODEL_name[@]}"; do

    for SECTOR in "${SECTOR_LIST[@]}"; do    
        echo "Running with SECTOR = ${SECTOR} | MODEL_name = ${MODEL}"
        python run.py \
            --task_name simulate \
            --exp_root_path SP500 \
            --sector ${SECTOR} \
            --model_type Diffusion \
            --model_name "${MODEL}" \
            --test_start_year 2022 \
            --total_test_months 36 \
            --sliding_test_months 12 \
            --train_months 12 \
            --noise_input_size 3 \
            --num_simulations 50 \
            --num_repeats 1 \
            --beta_start 0.0001 \
            --beta_end 0.02 \
            --T_diffusion 200 
        
         echo "Finished running with SECTOR = ${SECTOR} | MODEL_name = ${MODEL}"
    done
done