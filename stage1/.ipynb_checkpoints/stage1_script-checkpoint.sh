#!/bin/bash

set -e
set -u

stage=1


# train a model

if [ $stage -eq 1 ]; then
outputdir='../../data_mic/stage1_preprocessed_18hz_5fold/UMAFall'
mkdir -p $outputdir

echo '=================================running stage 1================================='
python stage1_preprocessing_UMAFall.py --dataset_name 'UMAFall' --sensor_loc 'waist' --split_mode '5fold' | tee $outputdir/stage1_logs.txt
python stage1_preprocessing_UMAFall.py --dataset_name 'UMAFall' --sensor_loc 'wrist' --split_mode '5fold' | tee $outputdir/stage1_logs.txt
python stage1_preprocessing_UMAFall.py --dataset_name 'UMAFall' --sensor_loc 'leg' --split_mode '5fold' | tee $outputdir/stage1_logs.txt
python stage1_preprocessing_UMAFall.py --dataset_name 'UMAFall' --sensor_loc 'chest' --split_mode '5fold' | tee $outputdir/stage1_logs.txt
python stage1_preprocessing_UMAFall.py --dataset_name 'UMAFall' --sensor_loc 'ankle' --split_mode '5fold' | tee $outputdir/stage1_logs.txt

# userjson='seizures_excel_0531.json'
# outputdir='../../../data/stage1/EpiWatch_sz'
# mkdir -p $outputdir
# python EpiWatch_sz_tests.py | $outputdir/tests_logs.txt


echo '=================================testing stage 1================================='

fi