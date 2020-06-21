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
training_params_file='training_params_list_HPsearch.json'

inputdir='../../data_mic/stage1/preprocessed_18hz_5fold'
outputdir='../../data_mic/stage2/modeloutput_18hz_5fold_UMAFall_cross-pos_loss_earlystop_HPsearch'
mkdir -p $outputdir

# tasks_list='UMAFall_chest-UMAFall_wrist UMAFall_chest-UMAFall_waist UMAFall_chest-UMAFall_ankle UMAFall_wrist-UMAFall_chest UMAFall_wrist-UMAFall_waist UMAFall_wrist-UMAFall_ankle UMAFall_waist-UMAFall_chest UMAFall_waist-UMAFall_wrist UMAFall_waist-UMAFall_ankle UMAFall_ankle-UMAFall_chest UMAFall_ankle-UMAFall_wrist UMAFall_ankle-UMAFall_waist'

# # red
# tasks_list='UMAFall_chest-UMAFall_ankle UMAFall_wrist-UMAFall_ankle UMAFall_waist-UMAFall_ankle UMAFall_ankle-UMAFall_chest UMAFall_ankle-UMAFall_wrist UMAFall_ankle-UMAFall_waist'
# orange
# tasks_list='UMAFall_chest-UMAFall_wrist UMAFall_wrist-UMAFall_chest UMAFall_wrist-UMAFall_waist UMAFall_waist-UMAFall_chest UMAFall_waist-UMAFall_wrist'
# the rest
tasks_list='UMAFall_chest-UMAFall_waist'

echo '=================================running stage 2 [early stop at lowest total loss for each pos combination]================================='


for tasks in $tasks_list

do
  python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt $show_diagnosis_plt --use_WeightedRandomSampler $use_WeightedRandomSampler --tasks_list $tasks | tee $outputdir/stage2_logs_UMA_pos.txt

done

echo '=================================testing stage 1================================='

fi