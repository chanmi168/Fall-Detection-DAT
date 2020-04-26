#!/bin/bash

set -e
set -u

stage=2


# train a model

if [ $stage -eq 2 ]; then
echo '=================================running stage 2================================='

extractor_type='CNN'
num_epochs=15
CV_n=5
rep_n=5
# cuda=2
inputdir='../../data_mic/stage1_preprocessed_NormalforAllAxes_18hz_5fold'
outputdir='../../data_mic/stage2_modeloutput_NormalforAllAxes_18hz_5fold_UPFall_pos'
mkdir -p $outputdir


python stage2_DANN_HPsearch_rep.py --input_folder $inputdir --output_folder $outputdir --extractor_type $extractor_type --num_epochs $num_epochs --CV_n $CV_n --rep_n $rep_n --show_diagnosis_plt 'True' --tasks_list 'UPFall_neck-UPFall_wrist UPFall_neck-UPFall_belt UPFall_neck-UPFall_rightpocket UPFall_neck-UPFall_ankle UPFall_wrist-UPFall_neck UPFall_wrist-UPFall_belt UPFall_wrist-UPFall_rightpocket UPFall_wrist-UPFall_ankle UPFall_belt-UPFall_neck UPFall_belt-UPFall_wrist UPFall_belt-UPFall_rightpocket UPFall_belt-UPFall_ankle UPFall_rightpocket-UPFall_neck UPFall_rightpocket-UPFall_wrist UPFall_rightpocket-UPFall_belt UPFall_rightpocket-UPFall_ankle UPFall_ankle-UPFall_neck UPFall_ankle-UPFall_wrist UPFall_ankle-UPFall_belt UPFall_ankle-UPFall_rightpocket' | tee $outputdir/stage2_logs_UPFall_pos.txt


echo '=================================testing stage 1================================='

fi