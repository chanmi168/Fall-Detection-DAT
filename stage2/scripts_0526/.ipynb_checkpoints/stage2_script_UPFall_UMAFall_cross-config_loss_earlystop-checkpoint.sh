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
show_diagnosis_plt='True'
use_WeightedRandomSampler='True'
training_params_file='training_params_list_fixed.json'

inputdir='../../data_mic/stage1/preprocessed_18hz_5fold'
outputdir='../../data_mic/stage2/modeloutput_18hz_5fold_UPFall_UMAFall_cross-config_loss_earlystop'
mkdir -p $outputdir

echo '=================================running stage 2 [early stop at lowest total loss]================================='

python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt $show_diagnosis_plt --use_WeightedRandomSampler $use_WeightedRandomSampler --tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' | tee $outputdir/stage2_logs_UMA-UP_config.txt

python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt $show_diagnosis_plt --use_WeightedRandomSampler $use_WeightedRandomSampler --tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' | tee $outputdir/stage2_logs_UP-UMA_config.txt


echo '=================================testing stage 1================================='

fi