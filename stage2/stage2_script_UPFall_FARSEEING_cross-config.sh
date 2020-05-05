#!/bin/bash

set -e
set -u

stage=2


# train a model

if [ $stage -eq 2 ]; then
extractor_type='CNN'
num_epochs=15
CV_n=5
rep_n=5
# training_params_file='training_params_list_v1.json'
training_params_file='training_params_list_fixed.json'


inputdir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_5fold'
outputdir='../../data_mic/stage2/modeloutput_NormalforAllAxes_18hz_5fold_UPFall_FARSEEING_cross-config'

mkdir -p $outputdir

echo '=================================running stage 2================================='

python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --tasks_list 'UPFall_rightpocket-FARSEEING_thigh UPFall_belt-FARSEEING_lowback' | tee $outputdir/stage2_logs_UPFall-FARSEEING_config.txt

# python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --tasks_list 'FARSEEING_thigh-UPFall_rightpocket FARSEEING_lowback-UPFall_belt' | tee $outputdir/stage2_logs_UP-UMA_config.txt


echo '=================================testing stage 1================================='

fi