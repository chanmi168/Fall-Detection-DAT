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

echo '=================================running UMAFall_UPFall_cross_config [dropout, lambda 0.31] allHP================================='

inputdir='../../data_mic/stage2/modeloutput_18hz_5fold_UPFall_UMAFall_cross-config_dropout0_5_lambda0_31'
outputdir='../../data_mic/stage3/UMAFall_UPFall_cross_config_diffCV_resample_dropout0_5_lambda0_31'
mkdir -p $outputdir

python stage3_model_eval_allHP.py \
--input_folder $inputdir \
--output_folder $outputdir \
--training_params_file $training_params_file \
--variable_name $variable_name \
--debug_F1 $debug_F1 \
--tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' \
| tee $outputdir/stage3_UMAFall_UPFall_logs.txt

python stage3_model_eval_allHP.py \
--input_folder $inputdir \
--output_folder $outputdir \
--training_params_file $training_params_file \
--variable_name $variable_name \
--debug_F1 $debug_F1 \
--tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' \
| tee $outputdir/stage3_UPFall_UMAFall_logs.txt


# echo '=================================running UMAFall_UPFall_cross_config [HPF, deeper] allHP================================='

# inputdir='../../data_mic/stage2/modeloutput_WithoutNormal_18hz_5fold_UPFall_UMAFall_cross-config_diffCV_deeper'
# outputdir='../../data_mic/stage3/UMAFall_UPFall_cross_config_diffCV_HPF_deeper'
# mkdir -p $outputdir


# python stage3_model_eval_allHP.py \
# --input_folder $inputdir \
# --output_folder $outputdir \
# --training_params_file $training_params_file \
# --variable_name $variable_name \
# --tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' \
# | tee $outputdir/stage3_UMAFall_UPFall_logs.txt

# python stage3_model_eval_allHP.py \
# --input_folder $inputdir \
# --output_folder $outputdir \
# --training_params_file $training_params_file \
# --variable_name $variable_name \
# --tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' \
# | tee $outputdir/stage3_UPFall_UMAFall_logs.txt


echo '=================================testing stage 1================================='

fi


