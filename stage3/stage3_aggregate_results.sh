#!/bin/bash

set -e
set -u

stage=3


# train a model

if [ $stage -eq 3 ]; then

echo '=================================running stage 3================================='

echo '=================================running UPFall_UMAFall_cross_config================================='

inputdir='../../data_mic/stage2_modeloutput_18hz_5fold'
outputdir='../../data_mic/stage3/UPFall_UMAFall_cross_config'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
--tgt_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle' \
| tee $outputdir/stage3_logs.txt

outputdir='../../data_mic/stage3/UMAFall_UPFall_cross_config'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' \
--src_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
| tee $outputdir/stage3_logs.txt

echo '=================================running UPFall_UMAFall_cross_config_norm================================='

inputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold'
outputdir='../../data_mic/stage3/UPFall_UMAFall_cross_config_norm'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
--tgt_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankl' \
| tee $outputdir/stage3_logs.txt

outputdir='../../data_mic/stage3/UMAFall_UPFall_cross_config_norm'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' \
--src_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
| tee $outputdir/stage3_logs.txt


echo '=================================running UPFall_UMAFall_cross_config_HPF================================='

inputdir='../../data_mic/stage2_modeloutput_WithoutNormal_18hz_5fold_UPFall_UMAFall_cross-config'
outputdir='../../data_mic/stage3/UPFall_UMAFall_cross_config_HPF'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
--tgt_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle' \
| tee $outputdir/stage3_logs.txt

outputdir='../../data_mic/stage3/UMAFall_UPFall_cross_config_HPF'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle' \
--src_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
| tee $outputdir/stage3_logs.txt


echo '************************************************************************************************************'

echo '=================================running UPFall_SFDLA_cross_config================================='


inputdir='../../data_mic/stage2_modeloutput_18hz_5fold'
outputdir='../../data_mic/stage3/UPFall_SFDLA_cross_config'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-SFDLA_chest UPFall_wrist-SFDLA_wrist UPFall_belt-SFDLA_waist UPFall_rightpocket-SFDLA_thigh UPFall_ankle-SFDLA_ankle' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
--tgt_names 'SFDLA_chest SFDLA_wrist SFDLA_waist SFDLA_thigh SFDLA_ankle' \
| tee $outputdir/stage3_logs.txt

outputdir='../../data_mic/stage3/SFDLA_UPFall_cross_config'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'SFDLA_chest-UPFall_neck SFDLA_wrist-UPFall_wrist SFDLA_waist-UPFall_belt SFDLA_thigh-UPFall_rightpocket SFDLA_ankle-UPFall_ankle' \
--src_names 'SFDLA_chest SFDLA_wrist SFDLA_waist SFDLA_thigh SFDLA_ankle' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
| tee $outputdir/stage3_logs.txt


echo '=================================running UPFall_SFDLA_cross_config_norm================================='

inputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold'
outputdir='../../data_mic/stage3/SFDLA_UPFall_cross_config_norm'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'SFDLA_chest-UPFall_neck SFDLA_wrist-UPFall_wrist SFDLA_waist-UPFall_belt SFDLA_thigh-UPFall_rightpocket SFDLA_ankle-UPFall_ankle' \
--src_names 'SFDLA_chest SFDLA_wrist SFDLA_waist SFDLA_thigh SFDLA_ankle' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
| tee $outputdir/stage3_logs.txt

outputdir='../../data_mic/stage3/UPFall_SFDLA_cross_config_norm'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-SFDLA_chest UPFall_wrist-SFDLA_wrist UPFall_belt-SFDLA_waist UPFall_rightpocket-SFDLA_thigh UPFall_ankle-SFDLA_ankle' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle' \
--tgt_names 'SFDLA_chest SFDLA_wrist SFDLA_waist SFDLA_thigh SFDLA_ankle' \
| tee $outputdir/stage3_logs.txt




echo '=================================running UPFall_cross_pos_norm================================='

inputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold_UPFall_pos'
outputdir='../../data_mic/stage3/UPFall_cross_pos_norm'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-UPFall_wrist UPFall_neck-UPFall_belt UPFall_neck-UPFall_rightpocket UPFall_neck-UPFall_ankle UPFall_wrist-UPFall_neck UPFall_wrist-UPFall_belt UPFall_wrist-UPFall_rightpocket UPFall_wrist-UPFall_ankle UPFall_belt-UPFall_neck UPFall_belt-UPFall_wrist UPFall_belt-UPFall_rightpocket UPFall_belt-UPFall_ankle UPFall_rightpocket-UPFall_neck UPFall_rightpocket-UPFall_wrist UPFall_rightpocket-UPFall_belt UPFall_rightpocket-UPFall_ankle UPFall_ankle-UPFall_neck UPFall_ankle-UPFall_wrist UPFall_ankle-UPFall_belt UPFall_ankle-UPFall_rightpocket' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_ankle UPFall_rightpocket' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_ankle UPFall_rightpocket' \
| tee $outputdir/stage3_logs.txt

echo '=================================running UPFall_cross_pos_HPF================================='

inputdir='../../data_mic/stage2_modeloutput_WithoutNormal_18hz_5fold_UPFall_pos'
outputdir='../../data_mic/stage3/UPFall_cross_pos_HPF'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UPFall_neck-UPFall_wrist UPFall_neck-UPFall_belt UPFall_neck-UPFall_rightpocket UPFall_neck-UPFall_ankle UPFall_wrist-UPFall_neck UPFall_wrist-UPFall_belt UPFall_wrist-UPFall_rightpocket UPFall_wrist-UPFall_ankle UPFall_belt-UPFall_neck UPFall_belt-UPFall_wrist UPFall_belt-UPFall_rightpocket UPFall_belt-UPFall_ankle UPFall_rightpocket-UPFall_neck UPFall_rightpocket-UPFall_wrist UPFall_rightpocket-UPFall_belt UPFall_rightpocket-UPFall_ankle UPFall_ankle-UPFall_neck UPFall_ankle-UPFall_wrist UPFall_ankle-UPFall_belt UPFall_ankle-UPFall_rightpocket' \
--src_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_ankle UPFall_rightpocket' \
--tgt_names 'UPFall_neck UPFall_wrist UPFall_belt UPFall_ankle UPFall_rightpocket' \
| tee $outputdir/stage3_logs.txt

echo '=================================running UMAFall_cross_pos_norm================================='

inputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold_UMAFall_pos'
outputdir='../../data_mic/stage3/UMAFall_cross_pos_norm'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UMAFall_chest-UMAFall_wrist UMAFall_chest-UMAFall_waist UMAFall_chest-UMAFall_ankle UMAFall_wrist-UMAFall_chest UMAFall_wrist-UMAFall_waist UMAFall_wrist-UMAFall_ankle UMAFall_waist-UMAFall_chest UMAFall_waist-UMAFall_wrist UMAFall_waist-UMAFall_ankle UMAFall_ankle-UMAFall_chest UMAFall_ankle-UMAFall_wrist UMAFall_ankle-UMAFall_waist' \
--src_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_ankle' \
--tgt_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_ankle' \
| tee $outputdir/stage3_logs.txt

echo '=================================running UMAFall_cross_pos_HPF================================='
inputdir='../../data_mic/stage2_modeloutput_WithoutNormal_18hz_5fold_UMAFall_pos'
outputdir='../../data_mic/stage3/UMAFall_cross_pos_HPF'
mkdir -p $outputdir

python stage3_model_eval.py \
--input_folder $inputdir \
--output_folder $outputdir \
--tasks_list 'UMAFall_chest-UMAFall_wrist UMAFall_chest-UMAFall_waist UMAFall_chest-UMAFall_ankle UMAFall_wrist-UMAFall_chest UMAFall_wrist-UMAFall_waist UMAFall_wrist-UMAFall_ankle UMAFall_waist-UMAFall_chest UMAFall_waist-UMAFall_wrist UMAFall_waist-UMAFall_ankle UMAFall_ankle-UMAFall_chest UMAFall_ankle-UMAFall_wrist UMAFall_ankle-UMAFall_waist' \
--src_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_ankle' \
--tgt_names 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_ankle' \
| tee $outputdir/stage3_logs.txt

echo '************************************************************************************************************'


echo '=================================testing stage 1================================='

fi


