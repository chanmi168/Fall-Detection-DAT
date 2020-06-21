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
outputdir='../../data_mic/stage2/modeloutput_18hz_5fold_UPFall_cross-pos_loss_earlystop_HPsearch'
mkdir -p $outputdir

# red
# tasks_list='UPFall_neck-UPFall_ankle UPFall_ankle-UPFall_neck UPFall_ankle-UPFall_wrist'
# orange
# tasks_list='UPFall_neck-UPFall_belt UPFall_neck_UPFall_rightpocket UPFall_wrist-UPFall_rightpocket UPFall_ankle-UPFall_rightpocket'
# the rest
# tasks_list='UPFall_neck-UPFall_wrist UPFall_wrist-UPFall_neck UPFall_wrist-UPFall_belt UPFall_belt-UPFall_neck UPFall_belt-UPFall_wrist UPFall_belt-UPFall_rightpocket UPFall_belt-UPFall_ankle UPFall_rightpocket-UPFall_neck UPFall_rightpocket-UPFall_wrist UPFall_rightpocket-UPFall_belt UPFall_rightpocket-UPFall_ankle UPFall_ankle-UPFall_belt'


tasks_list='UPFall_rightpocket-UPFall_neck UPFall_rightpocket-UPFall_wrist UPFall_rightpocket-UPFall_belt UPFall_rightpocket-UPFall_ankle UPFall_ankle-UPFall_belt'


echo '=================================running stage 2 [early stop at lowest total loss for each pos combination, HP search]================================='


for tasks in $tasks_list

do
  python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt $show_diagnosis_plt --use_WeightedRandomSampler $use_WeightedRandomSampler --tasks_list $tasks | tee $outputdir/stage2_logs_UP_pos.txt

done

echo '=================================testing stage 1================================='

fi