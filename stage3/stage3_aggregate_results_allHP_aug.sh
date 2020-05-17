#!/bin/bash

set -e
set -u

stage=3


# train a model

if [ $stage -eq 3 ]; then

echo '=================================running stage 3================================='

echo '=================================running UMAFall_UPFall_cross_config [aug] allHP================================='
training_params_file='training_params_list_fixed.json'
variable_name='channel_n'
inputdir='../../data_mic/stage2/modeloutput_WithoutNormal_18hz_5fold_aug_UPFall_UMAFall_cross-config_diffCV'
outputdir='../../data_mic/stage3/UMAFall_UPFall_cross_config_aug_diffCV'
mkdir -p $outputdir


python stage3_model_eval_allHP.py \
--input_folder $inputdir \
--output_folder $outputdir \
--training_params_file $training_params_file \
--variable_name $variable_name \
--tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' \
| tee $outputdir/stage3_UMAFall_UPFall_logs.txt

python stage3_model_eval_allHP.py \
--input_folder $inputdir \
--output_folder $outputdir \
--training_params_file $training_params_file \
--variable_name $variable_name \
--tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' \
| tee $outputdir/stage3_UPFall_UMAFall_logs.txt

echo '=================================testing stage 1================================='

fi


