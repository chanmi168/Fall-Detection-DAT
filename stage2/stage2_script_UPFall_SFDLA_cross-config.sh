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
# cuda=2
inputdir='../../data_mic/stage1/preprocessed_NormalforAllAxes_18hz_5fold'
outputdir='../../data_mic/stage2/modeloutput_NormalforAllAxes_18hz_5fold'
mkdir -p $outputdir

echo '=================================running stage 2================================='

# python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --cuda $cuda --tasks_list 'UPFall_neck-SFDLA_chest UPFall_wrist-SFDLA_wrist UPFall_belt-SFDLA_waist UPFall_rightpocket-SFDLA_thigh UPFall_ankle-SFDLA_ankle' | tee $outputdir/stage2_logs_UPFall-SFDLA_config.txt

python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --tasks_list 'SFDLA_chest-UPFall_neck SFDLA_wrist-UPFall_wrist SFDLA_waist-UPFall_belt SFDLA_thigh-UPFall_rightpocket SFDLA_ankle-UPFall_ankle' | tee $outputdir/stage2_logs_SFDLA-UPFall_config.txt

echo '=================================testing stage 1================================='

fi