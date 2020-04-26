#!/bin/bash

set -e
set -u

stage=1


# train a model

if [ $stage -eq 1 ]; then
# outputdir='../../data_mic/stage1_preprocessed_18hz_5fold'
# mkdir -p $outputdir
input_dir='../../Data/{}/ImpactWindow_Resample_NormalforAllAxes/18hz/{}/'
output_dir='../../data_mic/stage1_preprocessed_NormalforAllAxes_18hz_{}/{}/{}/'


echo '=================================running stage 1 UMAFall================================='
excluded_idx='1 3 9 10 12 19'

log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/UMAFall'
mkdir -p $log_dir

python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'waist' --split_mode '5fold' --i_seed 1 --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'wrist' --split_mode '5fold' --i_seed 1 --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'leg' --split_mode '5fold' --i_seed 1 --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'chest' --split_mode '5fold' --i_seed 1 --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UMAFall' --sensor_loc 'ankle' --split_mode '5fold' --i_seed 1 --excluded_idx "$excluded_idx" | tee  $log_dir/stage1_logs_UMAFall.txt


echo '=================================running stage 1 UPFall================================='
log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/UPFall'
mkdir -p $log_dir

python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'wrist' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_UPFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'rightpocket' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_UPFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'neck' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_UPFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'belt' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_UPFall.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'UPFall' --sensor_loc 'ankle' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_UPFall.txt

echo '=================================running stage 1 SFDLA================================='

log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/SFDLA'
mkdir -p $log_dir

python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'chest' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_SFDLA.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'wrist' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_SFDLA.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'waist' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_SFDLA.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'thigh' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_SFDLA.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'SFDLA' --sensor_loc 'ankle' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_SFDLA.txt

echo '=================================running stage 1 FARSEEING================================='

log_dir='../../data_mic/stage1_preprocessed_18hz_5fold/FARSEEING'
mkdir -p $log_dir

python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'FARSEEING' --sensor_loc 'lowback' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_FARSEEING.txt
python stage1_preprocessing_general.py --input_dir $input_dir --output_dir $output_dir --dataset_name 'FARSEEING' --sensor_loc 'thigh' --split_mode '5fold' --i_seed 1 | tee $log_dir/stage1_logs_FARSEEING.txt

echo '=================================testing stage 1================================='

fi