#!/usr/bin/env python
# coding: utf-8

# **stage2_DANN_HPsearch.ipynb**. This notebook attempts to search the hyperparameter that yields the optimal validation performance.
# 
# **Edit**<br/>
# 
# **TODO**<br/>

# # Import packages and get authenticated

# In[ ]:


# from google.colab import drive
# drive.mount('drive')


# In[2]:


import numpy as np

import pandas as pd
pd.set_option('display.max_columns', 500)
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
import os
import sys
sys.path.append('/content/drive/My Drive/中研院/repo/')
# sys.path.append('~/project_FDDAT/repo/')
sys.path.append('../') # add this line so Data and data are visible in this file
from os.path import expanduser
home = expanduser("~")

from falldetect.utilities import *
from falldetect.models import *
from falldetect.dataset_util import *
from falldetect.training_util import *
from falldetect.eval_util import *

import time
import datetime
from datetime import datetime
import json
import argparse

# Plotting
# checklist 1: comment inline, uncomment Agg
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )

# matplotlib.rc( 'savefig', transparent=True )

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:





# # Get user inputs
# In ipython notebook, these are hardcoded. In production python code, use parsers to provide these inputs

# In[3]:


parser = argparse.ArgumentParser(description='FD_DAT')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--training_params_file', metavar='training_params_file', help='training_params_file',
                    default='training_params_list_v0.json')
parser.add_argument('--extractor_type', metavar='extractor_type', help='extractor_type',
                    default='CNN')
parser.add_argument('--num_epochs', type=int, metavar='num_epochs', help='number of epochs',
                    default='5')
parser.add_argument('--CV_n', type=int, metavar='CV_n', help='CV folds',
                    default='2')
parser.add_argument('--rep_n', type=int, metavar='rep_n', help='number of repitition',
                    default='5')
# parser.add_argument('--cuda_i', type=int, metavar='cuda_i', help='cuda index',
#                     default='1')
parser.add_argument('--tasks_list', metavar='tasks_list', help='a list of all tasks',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')
parser.add_argument('--show_diagnosis_plt', metavar='show_diagnosis_plt', help='show diagnosis plt or not',
                    default='False')




# split_mode = 'LOO'
# split_mode = '5fold'

# checklist 2: comment first line, uncomment second line seizures_FN
# plt.style.use(['dark_background'])
# args = parser.parse_args(['--input_folder', '../../data_mic/stage1_preprocessed_WithoutNormal_18hz_5fold', 
#                           '--output_folder', '../../data_mic/stage2_modeloutput_WithoutNormal_18hz_5fold_test',
#                           '--training_params_file', 'training_params_list_v1.json',
#                           '--extractor_type', 'CNN',
#                           '--num_epochs', '10',
#                           '--CV_n', '2',
#                           '--rep_n', '2',
#                           '--show_diagnosis_plt', 'True',
#                           '--tasks_list', 'UPFall_rightpocket-UMAFall_leg UMAFall_leg-UPFall_rightpocket',])
#                           '--tasks_list', 'UMAFall_waist-UMAFall_wrist UPFall_wrist-UMAFall_ankle',])
                          
args = parser.parse_args()


# In[4]:


print(args)


# In[ ]:





# In[5]:


home_dir = home+'/project_FDDAT/'
input_folder = args.input_folder
output_folder = args.output_folder
training_params_file = args.training_params_file
extractor_type = args.extractor_type
num_epochs = args.num_epochs
CV_n = args.CV_n
rep_n = args.rep_n
show_diagnosis_plt = bool(args.show_diagnosis_plt)

with open('../../repo/falldetect/params.json') as json_file:
    falldetect_params = json.load(json_file)

cuda_i = falldetect_params['cuda_i']

tasks_list = []
for item in args.tasks_list.split(' '):
    tasks_list.append((item.split('-')[0], item.split('-')[1]))
    
inputdir = input_folder+'/'
outputdir = output_folder+'/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
# test_mode = 'test' in outputdir.split('/')[-2]
# test_mode = 'test' in training_params_file

device = torch.device('cuda:{}'.format(int(cuda_i)) if torch.cuda.is_available() else 'cpu')


# In[ ]:





# In[ ]:





# # new arch HP search

# In[6]:


# training_params_list = [
#   {
#     'HP_name': 'HP_i0',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 4,
#     'batch_size': 16,
#     'learning_rate': 0.01,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   },

#   {
#     'HP_name': 'HP_i1',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 4,
#     'batch_size': 4,
#     'learning_rate': 0.01,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   },

#   {
#     'HP_name': 'HP_i2',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 4,
#     'batch_size': 64,
#     'learning_rate': 0.01,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   },
    
#   {
#     'HP_name': 'HP_i3',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 16,
#     'batch_size': 4,
#     'learning_rate': 0.01,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   },

#   {
#     'HP_name': 'HP_i4',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 32,
#     'batch_size': 4,
#     'learning_rate': 0.01,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   },

#   {
#     'HP_name': 'HP_i5',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 4,
#     'batch_size': 4,
#     'learning_rate': 0.001,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   },

#   {
#     'HP_name': 'HP_i6',
#     'classes_n': 2,
#     'CV_n': CV_n,
#     'num_epochs': num_epochs,
#     'channel_n': 4,
#     'batch_size': 4,
#     'learning_rate': 0.0001,
#     'extractor_type': extractor_type,
# #     'device': device,
#     'dropout': 0.5,
#     'hiddenDim_f': 3,
#     'hiddenDim_y': 3,
#     'hiddenDim_d': 3,
#     'win_size': 18,
#     'win_stride': 6,
#     'step_n': 9,
#     'show_diagnosis_plt': show_diagnosis_plt,
#   }, ]

# with open('training_params_list.json', 'w') as fout:
#     json.dump(training_params_list, fout, indent=2)


# In[ ]:





# In[7]:


with open(training_params_file) as json_file:
    training_params_list = json.load(json_file)
    
for training_params in training_params_list:
    training_params['CV_n'] = CV_n
    training_params['num_epochs'] = num_epochs
    training_params['extractor_type'] = extractor_type
    training_params['device'] = device
    training_params['show_diagnosis_plt'] = show_diagnosis_plt


# In[ ]:





# In[ ]:





# In[8]:


# fine-tuning

df_metric_keys = ['df_acc', 'df_sensitivity', 'df_precision', 'df_F1']

for task_item in tasks_list:
    (src_name, tgt_name) = task_item

    task_outputdir = '{}{}_{}/'.format(outputdir, src_name, tgt_name)
    if not os.path.exists(task_outputdir):
        os.makedirs(task_outputdir)
    print('outputdir for stage2 {} output: {}'.format(task_item, task_outputdir))


    df_sample = pd.DataFrame('', index=['channel_n', 'batch_size', 'learning_rate', 
                                              'source', 'DANN', 'target', 'domain', 'time_elapsed', 'num_params'], columns=[])
    df_dict_agg_HP = dict( zip(df_metric_keys,[df_sample.copy(), df_sample.copy(), df_sample.copy(), df_sample.copy()]))


    # 1. try all HP
    for i, training_params in enumerate(training_params_list):
        df_dict = performance_table(src_name, tgt_name, training_params, inputdir, task_outputdir)
        for df_name in df_dict_agg_HP.keys():
            print('show', df_name)
            df_dict_agg_HP[df_name][training_params['HP_name']] = df_dict[df_name].copy()
#             display(df_dict_agg_HP[df_name])
        
    # 2. agg all HP
    
    if training_params_file=='training_params_list_test.json':
        pass
    elif training_params_file=='training_params_list_v1.json':
        pass
    elif training_params_file=='training_params_list_v0.json':
        for df_name in df_dict_agg_HP.keys():
            df_dict_agg_HP[df_name]['HP_i3_1'] = df_dict_agg_HP[df_name]['HP_i1']
            df_dict_agg_HP[df_name]['HP_i5_1'] = df_dict_agg_HP[df_name]['HP_i1']
    
    # 3. run optimal param for a task (based on dann sens.)
    training_params_optimal = training_params.copy()
    training_params_optimal['HP_name'] = 'HP_optimal'
    if training_params_file=='training_params_list_test.json':
        training_params_optimal['batch_size'] = 64
        training_params_optimal['channel_n'] = 4
        training_params_optimal['learning_rate'] = 0.01
    elif training_params_file=='training_params_list_v1.json':
        channel_n_optimal = get_optimal_v1(df_dict_agg_HP['df_sensitivity'])
        training_params_optimal['channel_n'] = channel_n_optimal
    elif training_params_file=='training_params_list_v0.json':
        batch_size_optimal, channel_n_optimal, learning_rate_optimal = get_optimal_v0(df_dict_agg_HP['df_sensitivity'])
        training_params_optimal['batch_size'] = batch_size_optimal
        training_params_optimal['channel_n'] = channel_n_optimal
        training_params_optimal['learning_rate'] = learning_rate_optimal

    
    df_dict = performance_table(src_name, tgt_name, training_params_optimal, inputdir, task_outputdir)

    for df_name in df_dict_agg_HP.keys():
        df_dict_agg_HP[df_name][training_params_optimal['HP_name']] = df_dict[df_name].copy()
    
    if training_params_file=='training_params_list_test.json':
        for df_name in df_dict_agg_HP.keys():
            df_dict_agg_HP[df_name] = df_dict_agg_HP[df_name][['test_i0','test_i1']]
    elif training_params_file=='training_params_list_v1.json':
            df_dict_agg_HP[df_name] = df_dict_agg_HP[df_name][['HP_i0','HP_i1','HP_i2','HP_i3','HP_i4','HP_optimal']]
    elif training_params_file=='training_params_list_v0.json':
        for df_name in df_dict_agg_HP.keys():
            df_dict_agg_HP[df_name] = df_dict_agg_HP[df_name][['HP_i0','HP_i1','HP_i2','HP_i3','HP_i3_1','HP_i4','HP_i5','HP_i5_1','HP_i6','HP_optimal']]

    # 4. store and display HP df_performance_table_agg

    df_outputdir = task_outputdir+'HP_search/'
    if not os.path.exists(df_outputdir):
        os.makedirs(df_outputdir)
    print('HP df_performance_table_agg saved at', df_outputdir)

    for df_name in df_dict_agg_HP.keys():
        df_dict_agg_HP[df_name].to_csv(df_outputdir+'df_performance_table_agg_HP_{}.csv'.format(df_name.split('_')[1]), encoding='utf-8')

    # Serialize data into file:
    json.dump({key:val for key, val in training_params_optimal.items() if key != 'device'}, open(df_outputdir+'training_params_optimal.json', 'w'))

    for df_name in df_dict_agg_HP.keys():
        print('show', df_name)
        display(df_dict_agg_HP[df_name])
        
    # 5. run rep experiments
    
    df_sample = pd.DataFrame('', index=['channel_n', 'batch_size', 'learning_rate', 
                                        'source', 'DANN', 'target', 'domain', 'time_elapsed', 'num_params'], columns=[])
    df_dict_agg_rep = dict( zip(df_metric_keys,[df_sample.copy(), df_sample.copy(), df_sample.copy(), df_sample.copy()]))

    for i in range(0,rep_n):
        df_dict = performance_table(src_name, tgt_name, training_params_optimal, inputdir, task_outputdir)
        for df_name in df_dict_agg_rep.keys():
            df_dict_agg_rep[df_name]['rep_i{}'.format(i)] = df_dict[df_name].copy()

    df_outputdir = task_outputdir+'repetitive_results/'
    if not os.path.exists(df_outputdir):
        os.makedirs(df_outputdir)
    print('df_performance_table_agg_rep saved at', df_outputdir)

    for df_name in df_dict_agg_rep.keys():
        df_dict_agg_rep[df_name] = get_rep_stats(df_dict_agg_rep[df_name], rep_n)
        df_dict_agg_rep[df_name].to_csv(df_outputdir+'df_performance_table_agg_rep_{}.csv'.format(df_name.split('_')[1]), encoding='utf-8')

    # Serialize data into file:
    json.dump({key:val for key, val in training_params_optimal.items() if key != 'device'}, open(df_outputdir+'training_params_optimal.json', 'w'))

    for df_name in df_dict_agg_rep.keys():
        print('show', df_name)
        display(df_dict_agg_rep[df_name])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




