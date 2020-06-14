#!/bin/bash

set -e
set -u

stage=3


# train a model

if [ $stage -eq 3 ]; then

echo '=================================running stage 3================================='
training_params_file='training_params_list_fixed.json'
variable_name='HP_name'
debug_F1='True'
rep_n=5

echo '=================================running UMAFall_cross_pos [earlystop at lowest total loss, HP fixed] allHP================================='

inputdir='../../data_mic/stage2/modeloutput_18hz_5fold_UMAFall_cross-pos_loss_earlystop'
outputdir='../../data_mic/stage3/UMAFall_cross-pos_loss_earlystop'
mkdir -p $outputdir

python stage3_model_eval_allHP.py \
--input_folder $inputdir \
--output_folder $outputdir \
--training_params_file $training_params_file \
--variable_name $variable_name \
--debug_F1 $debug_F1 \
--tasks_list 'UMAFall_chest-UMAFall_wrist UMAFall_chest-UMAFall_waist UMAFall_chest-UMAFall_ankle UMAFall_wrist-UMAFall_chest UMAFall_wrist-UMAFall_waist UMAFall_wrist-UMAFall_ankle UMAFall_waist-UMAFall_chest UMAFall_waist-UMAFall_wrist UMAFall_waist-UMAFall_ankle UMAFall_ankle-UMAFall_chest UMAFall_ankle-UMAFall_wrist UMAFall_ankle-UMAFall_waist' \
| tee $outputdir/stage3_UMAFall_UPFall_logs.txt

echo '=================================testing stage 1================================='

fi


