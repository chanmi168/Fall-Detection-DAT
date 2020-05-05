#!/bin/bash

set -e
set -u

stage=1


# train a model

if [ $stage -eq 1 ]; then
split_mode='5fold'
i_seed=1
# outputdir='../../data_mic/stage1_preprocessed_18hz_5fold'
# mkdir -p $outputdir

# input_dir='../../Data/{}/ImpactWindow_Resample/18hz/{}/'
# output_dir='../../data_mic/stage1/preprocessed_18hz_{}/{}/{}/'

input_dir='../../Data/{}/ImpactWindow_Resample_NormalforAllAxes/18hz/{}/'
output_dir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_{}/{}/{}/'

# input_dir='../../Data/{}/ImpactWindow_Resample_WithoutNormal/18hz/{}/'
# output_dir='../../data_mic/stage1/preprocessed_WithoutNormal_18hz_{}/{}/{}/'

# echo '=================================running stage 1 UMAFall================================='
# excluded_idx='1 3 9 10 12 19'

# # log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/UMAFall'
# log_dir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_5fold/UMAFall'
# # log_dir='../../data_mic/stage1_preprocessed_WithoutNormal_18hz_5fold/UMAFall'
# mkdir -p $log_dir

# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'waist' --split_mode $split_mode --i_seed $i_seed --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_waist.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'wrist' --split_mode $split_mode --i_seed $i_seed --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_wrist.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'leg' --split_mode $split_mode --i_seed $i_seed --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_leg.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'chest' --split_mode $split_mode --i_seed $i_seed --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_chest.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'ankle' --split_mode $split_mode --i_seed $i_seed --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall_ankle.txt


# echo '=================================running stage 1 UPFall================================='
# # log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/UPFall'
# log_dir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_5fold/UPFall'
# # log_dir='../../data_mic/stage1_preprocessed_WithoutNormal_18hz_5fold/UPFall'
# mkdir -p $log_dir

# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'wrist' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_UPFall_wrist.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'rightpocket' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_UPFall_rightpocket.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'neck' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_UPFall_neck.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'belt' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_UPFall_belt.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'ankle' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_UPFall_ankle.txt

# echo '=================================running stage 1 SFDLA================================='
# # log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/SFDLA'
# log_dir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_5fold/SFDLA'
# # log_dir='../../data_mic/stage1_preprocessed_WithoutNormal_18hz_5fold/SFDLA'
# mkdir -p $log_dir

# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'chest' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_SFDLA_chest.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'wrist' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_SFDLA_wrist.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'waist' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_SFDLA_waist.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'thigh' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_SFDLA_thigh.txt
# python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'ankle' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_SFDLA_ankle.txt

echo '=================================running stage 1 FARSEEING================================='
# log_dir='../../data_mic/stage1/preprocessed_18hz_5fold/FARSEEING'
log_dir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_5fold/FARSEEING'
mkdir -p $log_dir

python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'FARSEEING' --sensor_loc 'lowback' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_FARSEEING_lowback.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'FARSEEING' --sensor_loc 'thigh' --split_mode $split_mode --i_seed $i_seed | tee $log_dir/stage1_logs_FARSEEING_thigh.txt


echo '=================================testing stage 1================================='

fi