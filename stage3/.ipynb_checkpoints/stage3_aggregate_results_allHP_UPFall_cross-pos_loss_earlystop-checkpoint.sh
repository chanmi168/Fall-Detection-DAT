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

echo '=================================running UPFall_cross_pos [earlystop at lowest total loss, HP fixed] allHP================================='

inputdir='../../data_mic/stage2/modeloutput_18hz_5fold_UPFall_cross-pos_loss_earlystop'
outputdir='../../data_mic/stage3/UPFall_cross-pos_loss_earlystop'
mkdir -p $outputdir

python stage3_model_eval_allHP.py \
--input_folder $inputdir \
--output_folder $outputdir \
--training_params_file $training_params_file \
--variable_name $variable_name \
--debug_F1 $debug_F1 \
--tasks_list 'UPFall_neck-UPFall_wrist UPFall_neck-UPFall_belt UPFall_neck-UPFall_rightpocket UPFall_neck-UPFall_ankle UPFall_wrist-UPFall_neck UPFall_wrist-UPFall_belt UPFall_wrist-UPFall_rightpocket UPFall_wrist-UPFall_ankle UPFall_belt-UPFall_neck UPFall_belt-UPFall_wrist UPFall_belt-UPFall_rightpocket UPFall_belt-UPFall_ankle UPFall_rightpocket-UPFall_neck UPFall_rightpocket-UPFall_wrist UPFall_rightpocket-UPFall_belt UPFall_rightpocket-UPFall_ankle UPFall_ankle-UPFall_neck UPFall_ankle-UPFall_wrist UPFall_ankle-UPFall_belt UPFall_ankle-UPFall_rightpocket' \
| tee $outputdir/stage3_UMAFall_UPFall_logs.txt

echo '=================================testing stage 1================================='

fi


