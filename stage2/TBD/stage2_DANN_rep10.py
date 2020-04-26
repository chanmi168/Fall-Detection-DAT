#!/usr/bin/env python
# coding: utf-8

# **stage2_DANN_rep_n**. This notebook perform DANN training on optimal hyperparameters for rep_n times.
# 
# **Edit**<br/>
# 
# **TODO**<br/>

# # Import packages and get authenticated

# In[1]:


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

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:





# In[ ]:





# # Get user inputs
# In ipython notebook, these are hardcoded. In production python code, use parsers to provide these inputs

# In[3]:


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
                    default='2')
parser.add_argument('--rep_n', type=int, metavar='rep_n', help='number of repitition',
                    default='5')
parser.add_argument('--tasks_list', metavar='tasks_list', help='a list of all tasks',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')



# split_mode = 'LOO'
# split_mode = '5fold'

# checklist 2: comment first line, uncomment second line seizures_FN
# args = parser.parse_args(['--input_folder', 'stage1_preprocessed_18hz_5fold', 
#                           '--output_folder', 'stage2_modeloutput_18hz_5fold',
#                           '--extractor_type', 'CNN',
#                           '--num_epochs', '2',
#                           '--CV_n', '2',
#                           '--rep_n', '2',
#                           '--tasks_list', 'UMAFall_waist-UPFall_belt UPFall_wrist-UMAFall_ankle',])
                          
args = parser.parse_args()


# In[4]:


home_dir = home+'/project_FDDAT/'
input_folder = args.input_folder
output_folder = args.output_folder
extractor_type = args.extractor_type
num_epochs = args.num_epochs
CV_n = args.CV_n
rep_n = args.rep_n

tasks_list = []
for item in args.tasks_list.split(' '):
    tasks_list.append((item.split('-')[0], item.split('-')[1]))
    
inputdir = home_dir + 'data_mic/{}/'.format(input_folder)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# In[ ]:





# # Repeat 10 times experiment

# In[5]:


#  


# In[6]:


def get_rep_stats_2(df_performance_table_agg, rep_n):
    df_performance_table_agg = df_performance_table_agg[['rep_i0', 'rep_i1']]

    df_acc = df_performance_table_agg.loc[ ['source', 'DANN', 'target', 'domain'] , ]
    df_params = df_performance_table_agg.loc[ ['channel_n', 'batch_size', 'learning_rate', 'time_elapsed', 'num_params'], ]

    # accs
    df_performance_table_all_mean = df_acc.applymap(get_mean)
    df_performance_table_means = df_performance_table_all_mean.mean(axis=1)
    df_performance_table_stds = df_performance_table_all_mean.std(axis=1)
    df_performance_table_all_mean['mean'] = df_performance_table_means
    df_performance_table_all_mean['std'] = df_performance_table_stds
    df_performance_table_all_mean['rep'] = df_performance_table_all_mean[['mean', 'std']].apply(lambda x : '{:.3f}±{:.3f}'.format(x[0],x[1]), axis=1)

    # params
    df_params_means = df_params.mean(axis=1)

    df_performance_table_agg['rep'] = ''
    df_performance_table_agg.loc[ ['source', 'DANN', 'target', 'domain'] , ['rep']] = df_performance_table_all_mean.loc[:, ['rep']]
    df_performance_table_agg.loc[ ['channel_n', 'batch_size', 'learning_rate', 'time_elapsed', 'num_params'] , ['rep']] = df_params_means

    return df_performance_table_agg




# In[7]:


# df_performance_table_agg
# df_performance_table_agg.loc[ ['source', 'DANN', 'target', 'domain'] , ].applymap(get_mean)
# df_acc.applymap(get_mean)


# In[8]:


tasks_params_list = [

    
  { 'HP_name': 'opitmal',
    'task': tasks_list[0],
    'classes_n': 2,
    'CV_n': CV_n,
    'num_epochs': num_epochs,
    'channel_n': 4,
    'batch_size': 4,
    'learning_rate': 0.0001,
    'extractor_type': extractor_type,
    'device': device}, 
    
] 


for tasks_params in tasks_params_list:

    (src_name, tgt_name) = tasks_params['task']

#     if 'rightpocket' in src_name or 'leg' in tgt_name or 'rightpocket' in tgt_name or 'leg' in src_name:
#         tasks_params['CV_n'] = 15
#     else:
#         tasks_params['CV_n'] = 17


    task_outputdir = home_dir + 'data_mic/{}/{}_{}/'.format(output_folder, src_name, tgt_name)

    if not os.path.exists(task_outputdir):
        os.makedirs(task_outputdir)
    print('outputdir for stage2 output:', task_outputdir)
    
    df_performance_table_agg = pd.DataFrame('', index=['channel_n', 'batch_size', 'learning_rate', 
                                                      'source', 'DANN', 'target', 'domain', 'time_elapsed', 'num_params'], columns=[])

    
    for i in range(0,rep_n):
        df_performance_table = performance_table(src_name, tgt_name, tasks_params, inputdir, task_outputdir)
        df_performance_table_agg['rep_i{}'.format(i)] = df_performance_table

    df_outputdir = task_outputdir+'repetitive_results/'
    if not os.path.exists(df_outputdir):
        os.makedirs(df_outputdir)
    print('df_performance_table_rep_agg saved at', df_outputdir)

    # Serialize data into file:
    json.dump({key:val for key, val in tasks_params.items() if key != 'device'}, open(df_outputdir+'optimal_training_params.json', 'w'))

    df_performance_table_agg = get_rep_stats_2(df_performance_table_agg, rep_n)
    df_performance_table_agg.to_csv(df_outputdir+'df_performance_table_rep{}_agg.csv'.format(rep_n, i), encoding='utf-8')

    display(df_performance_table_agg)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




