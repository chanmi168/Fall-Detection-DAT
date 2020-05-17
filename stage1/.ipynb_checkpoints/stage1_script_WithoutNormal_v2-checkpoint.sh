#!/bin/bash

set -e
set -u

stage=1


# train a model
if [ $stage -eq 1 ]; then
split_mode='5fold'
# i_seed='1 2 3 4 5 6 7 8 9 10'
i_seed='1'
rep_n='10'

input_dir='../../Data/{}/ImpactWindow_Resample_WithoutNormal/18hz/{}/{}/'
output_dir='../../data_mic/stage1/preprocessed_WithoutNormal_18hz_{}_aug/{}/{}/'

echo '=================================running stage 1 UMAFall================================='
excluded_idx='1 3 9 10 12 19'

log_dir='../../data_mic/stage1/preprocessed_WithoutNormal_18hz_5fold_aug/UMAFall'
mkdir -p $log_dir

python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'waist' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_waist.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'wrist' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_wrist.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'leg' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_leg.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'chest' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_chest.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'ankle' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_ankle.txt


echo '=================================running stage 1 UPFall================================='
log_dir='../../data_mic/stage1/preprocessed_WithoutNormal_18hz_5fold_aug/UPFall'
mkdir -p $log_dir

python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'wrist' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" | tee $log_dir/stage1_logs_UPFall_wrist.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'rightpocket' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" | tee $log_dir/stage1_logs_UPFall_rightpocket.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'neck' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" | tee $log_dir/stage1_logs_UPFall_neck.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'belt' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" | tee $log_dir/stage1_logs_UPFall_belt.txt
python stage1_preprocessing_general_v2.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'ankle' --split_mode $split_mode --i_seed "$i_seed" --rep_n "$rep_n" | tee $log_dir/stage1_logs_UPFall_ankle.txt


echo '=================================testing stage 1================================='

fi
