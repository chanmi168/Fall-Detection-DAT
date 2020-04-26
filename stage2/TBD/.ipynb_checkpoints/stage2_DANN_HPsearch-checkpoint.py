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


# In[1]:


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

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:





# # Get user inputs
# In ipython notebook, these are hardcoded. In production python code, use parsers to provide these inputs

# In[3]:


# tasks_list = [('UMAFall_waist', 'UPFall_belt'), ('UMAFall_waist', 'UPFall_belt')]

# # extractor_type = 'CNN'
# # num_epochs = 2
# # CV_n = 2
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# In[4]:


parser = argparse.ArgumentParser(description='FD_DAT')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--extractor_type', metavar='extractor_type', help='extractor_type',
                    default='CNN')
parser.add_argument('--num_epochs', type=int, metavar='num_epochs', help='number of epochs',
                    default='5')
parser.add_argument('--CV_n', type=int, metavar='CV_n', help='CV folds',
                    default='../')
parser.add_argument('--tasks_list', metavar='tasks_list', help='a list of all tasks',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')



# split_mode = 'LOO'
# split_mode = '5fold'

# checklist 2: comment first line, uncomment second line seizures_FN
args = parser.parse_args(['--input_folder', 'stage1_preprocessed_18hz_5fold', 
                          '--output_folder', 'stage2_modeloutput_18hz_5fold',
                          '--extractor_type', 'CNN',
                          '--num_epochs', '2',
                          '--CV_n', '2',
                          '--tasks_list', 'UMAFall_waist-UPFall_belt UPFall_wrist-UMAFall_ankle',])
                          
# args = parser.parse_args()


# In[ ]:





# In[ ]:





# In[5]:


home_dir = home+'/project_FDDAT/'
input_folder = args.input_folder
output_folder = args.output_folder
extractor_type = args.extractor_type
num_epochs = args.num_epochs
CV_n = args.CV_n

tasks_list = []
for item in args.tasks_list.split(' '):
    tasks_list.append((item.split('-')[0], item.split('-')[1]))
    
inputdir = home_dir + 'data_mic/{}/'.format(input_folder)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# In[ ]:





# In[ ]:





# In[6]:


# CV_n = 17
# tasks_list = [('UMAFall_wrist', 'UPFall_neck'), ('UMAFall_wrist', 'UPFall_belt'), ('UMAFall_wrist', 'UPFall_ankle'),
#               ('UMAFall_waist', 'UPFall_neck'), ('UMAFall_waist', 'UPFall_wrist'), ('UMAFall_waist', 'UPFall_ankle'),
#               ('UMAFall_ankle', 'UPFall_neck'), ('UMAFall_ankle', 'UPFall_wrist'), ('UMAFall_ankle', 'UPFall_belt')]

# CV_n = 15
# tasks_list = [('UMAFall_wrist', 'UPFall_rightpocket'), ('UMAFall_waist', 'UPFall_rightpocket'), ('UMAFall_ankle', 'UPFall_rightpocket'),
#               ('UMAFall_leg', 'UPFall_neck'), ('UMAFall_leg', 'UPFall_wrist'), ('UMAFall_leg', 'UPFall_belt'), ('UMAFall_leg', 'UPFall_ankle')]


# # new arch HP search

# In[7]:


# def get_optimal(df_performance_table_agg):
#     df_performance_table_agg_temp = df_performance_table_agg.copy()

#     result = df_performance_table_agg_temp[['HP_i0','HP_i1','HP_i2']].sort_values(by='DANN', ascending=False, axis=1)
#     batch_size_optimal = result.loc['batch_size'][0]

#     result = df_performance_table_agg_temp[['HP_i3','HP_i3_1','HP_i4']].sort_values(by='DANN', ascending=False, axis=1)
#     channel_n_optimal = result.loc['channel_n'][0]

#     result = df_performance_table_agg_temp[['HP_i5','HP_i5_1','HP_i6']].sort_values(by='DANN', ascending=False, axis=1)
#     learning_rate_optimal = result.loc['learning_rate'][0]

#     return int(batch_size_optimal), int(channel_n_optimal), learning_rate_optimal


# In[8]:


training_params_list = [
  {
    'HP_name': 'HP_i0',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 4,
    'batch_size': 16,
    'learning_rate': 0.01,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  },

  {
    'HP_name': 'HP_i1',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 4,
    'batch_size': 4,
    'learning_rate': 0.01,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  },

  {
    'HP_name': 'HP_i2',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 4,
    'batch_size': 64,
    'learning_rate': 0.01,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  },
    
  {
    'HP_name': 'HP_i3',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 16,
    'batch_size': 4,
    'learning_rate': 0.01,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  },

  {
    'HP_name': 'HP_i4',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 32,
    'batch_size': 4,
    'learning_rate': 0.01,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  },

  {
    'HP_name': 'HP_i5',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 4,
    'batch_size': 4,
    'learning_rate': 0.001,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  },

  {
    'HP_name': 'HP_i6',
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 4,
    'batch_size': 4,
    'learning_rate': 0.0001,
    'extractor_type': extractor_type,
    'device': device,
    'dropout': 0.5,
    'hiddenDim_f': 3,
    'hiddenDim_y': 3,
    'hiddenDim_d': 3,
    'win_size': 18,
    'win_stride': 6,
    'step_n': 9,
  }, ]


# In[9]:


# fine-tuning

# tasks_list = [('UMAFall_waist', 'UPFall_belt'), ('UMAFall_waist', 'UPFall_belt')]

# extractor_type = 'CNN'
# num_epochs = 2
# CV_n = 2
# rep_n = 2
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')



for task_item in tasks_list:
    (src_name, tgt_name) = task_item

    task_outputdir = home_dir + 'data_mic/{}/{}_{}/'.format(output_folder, src_name, tgt_name)

    if not os.path.exists(task_outputdir):
        os.makedirs(task_outputdir)
    print('outputdir for stage2 {} output: {}'.format(task_item, task_outputdir))

    df_outputdir = task_outputdir+'HP_search/'
    if not os.path.exists(df_outputdir):
        os.makedirs(df_outputdir)
    print('HP df_performance_table_agg saved at', df_outputdir)

    df_performance_table_agg = pd.DataFrame('', index=['channel_n', 'batch_size', 'learning_rate', 
                                                  'source', 'DANN', 'target', 'domain', 'time_elapsed', 'num_params'], columns=[])

  # 1. try all HP
    for i, training_params in enumerate(training_params_list):
#         if 'rightpocket' in src_name or 'leg' in tgt_name or 'rightpocket' in tgt_name or 'leg' in src_name:
#             training_params['CV_n'] = 15
#         else:
#             training_params['CV_n'] = 17
        df_performance_table = performance_table(src_name, tgt_name, training_params, inputdir, task_outputdir)
        df_performance_table_agg[training_params['HP_name']] = df_performance_table


    # 2. agg all HP
    df_performance_table_agg['HP_i3_1'] = df_performance_table_agg['HP_i1']
    df_performance_table_agg['HP_i5_1'] = df_performance_table_agg['HP_i1']

    # 3 run optimal param for a task
    batch_size_optimal, channel_n_optimal, learning_rate_optimal = get_optimal(df_performance_table_agg)
    training_params_optimal = training_params.copy()
    training_params_optimal['HP_name'] = 'HP_optimal'
    training_params_optimal['batch_size'] = batch_size_optimal
    training_params_optimal['channel_n'] = channel_n_optimal
    training_params_optimal['learning_rate'] = learning_rate_optimal

    df_performance_table = performance_table(src_name, tgt_name, training_params_optimal, inputdir, task_outputdir)
    df_performance_table_agg[training_params_optimal['HP_name']] = df_performance_table

    df_performance_table_agg = df_performance_table_agg[['HP_i0','HP_i1','HP_i2','HP_i3','HP_i3_1','HP_i4','HP_i5','HP_i5_1','HP_i6','HP_optimal']]
    display(df_performance_table_agg)

    print('df_outputdir for stage2 df_performance_table_HP_agg:', df_outputdir)
    df_performance_table_agg.to_csv(df_outputdir+'df_performance_table_HP_agg.csv', encoding='utf-8')

    # Serialize data into file:
    json.dump({key:val for key, val in training_params.items() if key != 'device'}, open(df_outputdir+'optimal_training_params.json', 'w'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




