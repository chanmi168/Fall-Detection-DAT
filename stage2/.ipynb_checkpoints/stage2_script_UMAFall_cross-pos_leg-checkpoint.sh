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
training_params_file='training_params_list_v1.json'

# inputdir='../../data_mic/stage1_preprocessed_NormalforAllAxes_18hz_5fold'
# outputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold_UMAFall_pos'
inputdir='../../data_mic/stage1/preprocessed_WithoutNormal_18hz_5fold'
outputdir='../../data_mic/stage2/modeloutput_WithoutNormal_18hz_5fold_UMAFall_pos'
mkdir -p $outputdir

echo '=================================running stage 2================================='

python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --training_params_file $training_params_file --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --tasks_list 'UMAFall_leg-UMAFall_chest UMAFall_leg-UMAFall_wrist UMAFall_leg-UMAFall_waist UMAFall_leg-UMAFall_ankle UMAFall_chest-UMAFall_leg UMAFall_wrist-UMAFall_leg UMAFall_waist-UMAFall_leg UMAFall_ankle-UMAFall_leg' | tee $outputdir/stage2_logs_UMAFall_pos.txt

echo '=================================testing stage 1================================='

fi

