# #!/bin/bash
# # Define the list of train_months values
# TRAIN_MONTHS_LIST=(12 24 36)

# # Define the list of num_scenarios values
# NUM_SCENARIOS_LIST=($(seq 25 25 100))

# # =========================================================
# for TRAIN_MONTHS in "${TRAIN_MONTHS_LIST[@]}"; do
#     for NUM_SCENARIOS in "${NUM_SCENARIOS_LIST[@]}"; do

#         echo "Running with ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"        
#         python run.py \
#             --task_name Portfolio \
#             --bt_name Portfolio_Construction \
#             --exp_root_path SP500 \
#             --test_start_year 2022 \
#             --total_test_months 36 \
#             --sliding_test_months 12 \
#             --train_months ${TRAIN_MONTHS} \
#             --num_scenarios ${NUM_SCENARIOS}
        
#         echo "Finished running with ${TRAIN_MONTHS} | num_scenarios = ${NUM_SCENARIOS}"

#     done
# done

python run.py \
    --task_name Portfolio_strategy \
    --bt_name PlotResults \
    --exp_root_path SP500 \
    --test_start_year 2022 \
    --total_test_months 36 \
    --sliding_test_months 12
