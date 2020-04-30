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
inputdir='../../data_mic/stage1_preprocessed_NormalforAllAxes_18hz_5fold'
outputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold_SFDLA_pos'
mkdir -p $outputdir

echo '=================================running stage 2================================='

SFDLA_chest- SFDLA_wrist- SFDLA_waist- SFDLA_thigh- SFDLA_ankle-

python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file 'training_params_list_v1.json' --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --tasks_list \
'SFDLA_chest-SFDLA_wrist SFDLA_chest-SFDLA_waist SFDLA_chest-SFDLA_thigh SFDLA_chest-SFDLA_ankle \
SFDLA_wrist-SFDLA_chest SFDLA_wrist-SFDLA_waist SFDLA_wrist-SFDLA_thigh SFDLA_wrist-SFDLA_ankle \
SFDLA_waist-SFDLA_chest SFDLA_waist-SFDLA_wrist SFDLA_waist-SFDLA_thigh SFDLA_waist-SFDLA_ankle \
SFDLA_thigh-SFDLA_chest SFDLA_thigh-SFDLA_wrist SFDLA_thigh-SFDLA_waist SFDLA_thigh-SFDLA_ankle \
SFDLA_ankle-SFDLA_chest SFDLA_ankle-SFDLA_wrist SFDLA_ankle-SFDLA_waist SFDLA_ankle-SFDLA_thigh' | tee $outputdir/stage2_logs_UMAFall_pos.txt

echo '=================================testing stage 1================================='

fi