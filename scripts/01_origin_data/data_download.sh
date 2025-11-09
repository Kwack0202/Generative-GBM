#!/bin/bash
python run.py \
    --task_name data_download \
    --metadata ./datasets/metadata_info.csv \
    --exp_root_path SP500 \
    --start_day 2019-01-01 \
    --end_day 2025-01-01 \