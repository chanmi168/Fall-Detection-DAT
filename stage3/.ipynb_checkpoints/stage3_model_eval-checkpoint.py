#!/usr/bin/env python
# coding: utf-8

# **stage3_model_eval**. This notebook evaluates the trained model
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
import copy
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.options.display.float_format = "{:,.3f}".format

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


# # Get user inputs
# In ipython notebook, these are hardcoded. In production python code, use parsers to provide these inputs

# In[9]:


parser = argparse.ArgumentParser(description='FD_DAT')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
# parser.add_argument('--extractor_type', metavar='extractor_type', help='extractor_type',
#                     default='CNN')
# parser.add_argument('--num_epochs', type=int, metavar='num_epochs', help='number of epochs',
#                     default='5')
# parser.add_argument('--CV_n', type=int, metavar='CV_n', help='CV folds',
#                     default='2')
# parser.add_argument('--rep_n', type=int, metavar='rep_n', help='number of repitition',
#                     default='5')
parser.add_argument('--tasks_list', metavar='tasks_list', help='a list of all tasks',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')
parser.add_argument('--src_names', metavar='src_names', help='a list of src_names',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')
parser.add_argument('--tgt_names', metavar='tgt_names', help='a list of tgt_names',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')



# split_mode = 'LOO'
# split_mode = '5fold'

# checklist 2: comment first line, uncomment second line seizures_FN

# args = parser.parse_args(['--input_folder', '../../data_mic/stage2/modeloutput_WithoutNormal_18hz_5fold_UPFall_UMAFall_cross-config', 
#                           '--output_folder', '../../data_mic/stage3/UMAFall_UPFall_cross-config',
#                           '--tasks_list', 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle',
#                           '--src_names',  'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle',
#                           '--tgt_names',  'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle',
#                          ])
args = parser.parse_args()


# In[ ]:





# In[10]:


home_dir = home+'/project_FDDAT/'
input_folder = args.input_folder
output_folder = args.output_folder

tasks_list = []
for item in args.tasks_list.split(' '):
    tasks_list.append((item.split('-')[0], item.split('-')[1]))
    
src_domains = args.src_names.split(' ')
tgt_domains = args.tgt_names.split(' ')

inputdir = input_folder + '/'
outputdir = output_folder + '/'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


df_metric_keys = ['df_acc', 'df_sensitivity', 'df_precision', 'df_F1']
df_sample = pd.DataFrame(float('NaN'), columns=tgt_domains, index=src_domains)
df_sample = df_sample.loc[:,~df_sample.columns.duplicated()]
df_sample = df_sample.loc[~df_sample.index.duplicated(keep='first')]

df_dict_cross_config = dict( zip(df_metric_keys,[df_sample.copy(), df_sample.copy(), df_sample.copy(), df_sample.copy()]))

                                  
training_setting_list = ['source', 'DANN', 'target', 'improvement']
finetuned_results = dict( zip(training_setting_list,[copy.deepcopy(df_dict_cross_config), copy.deepcopy(df_dict_cross_config), copy.deepcopy(df_dict_cross_config), copy.deepcopy(df_dict_cross_config), ]))

# df_task_list = []
for task_item in tasks_list:
    for training_setting in training_setting_list:
        for metric_name in df_metric_keys:
            if training_setting == 'improvement':
                finetuned_results[training_setting][metric_name].at[src_name, tgt_name] = finetuned_results['DANN'][metric_name].at[src_name, tgt_name] - finetuned_results['source'][metric_name].at[src_name, tgt_name] 
                continue
            (src_name, tgt_name) = task_item
            df_task_inputdir = inputdir+src_name+'_'+tgt_name+'/repetitive_results/df_performance_table_agg_rep_{}.csv'.format(metric_name.split('_')[1])
            if os.path.isfile(df_task_inputdir):
                df_task = pd.read_csv(df_task_inputdir, index_col=0)[['rep_avg']].copy()
            else:
                continue
            df_task.rename(columns={'rep_avg':src_name+'_'+tgt_name}, inplace=True)

            finetuned_results[training_setting][metric_name].at[src_name, tgt_name] = float(df_task.copy().iloc[:, 0][training_setting].split('±')[0])

for training_setting in finetuned_results.keys():
    for metric_name in finetuned_results[training_setting].keys():
        print('training_setting: {}, metric_name: {}'.format(training_setting, metric_name))
        df_outputdir = '{}{}/'.format(outputdir, metric_name.split('_')[1])
        if not os.path.exists(df_outputdir):
            os.makedirs(df_outputdir)
        print('will export data to', df_outputdir)

        finetuned_results[training_setting][metric_name].round(5).to_csv(df_outputdir+'df_performance_table_agg_{}.csv'.format(training_setting), encoding='utf-8')
        display(finetuned_results[training_setting][metric_name])


# In[ ]:





# In[ ]:





# In[14]:


df_task_list = dict( zip(df_metric_keys,[[], [], [], []]))

for task_item in tasks_list:
    for metric_name in df_metric_keys:
        (src_name, tgt_name) = task_item
        df_task_inputdir = inputdir+src_name+'_'+tgt_name+'/repetitive_results/df_performance_table_agg_rep_{}.csv'.format(metric_name.split('_')[1])
        if os.path.isfile(df_task_inputdir):
            df_task = pd.read_csv(df_task_inputdir, index_col=0)[['rep_avg']].copy()
        else:
            continue
        df_task.rename(columns={'rep_avg':src_name+'_'+tgt_name}, inplace=True)
        
        improve_DANN = float(df_task.loc['DANN'].values[0].split('±')[0])-float(df_task.loc['source'].values[0].split('±')[0])
        improve_target = float(df_task.loc['target'].values[0].split('±')[0])-float(df_task.loc['source'].values[0].split('±')[0])
        df_task.loc['DANN'] = '{}({:+.3f})'.format(df_task.loc['DANN'].values[0], improve_DANN)
        df_task.loc['target'] = '{}({:+.3f})'.format(df_task.loc['target'].values[0], improve_target)
        
        if 'PAD_source' in df_task.index:
            improve_PAD_source = float(df_task.loc['PAD_DANN'].values[0].split('±')[0])-float(df_task.loc['PAD_source'].values[0].split('±')[0])
            df_task.loc['PAD_DANN'] = '{}({:+.3f})'.format(df_task.loc['PAD_DANN'].values[0], improve_PAD_source)
        df_task_list[metric_name].append(df_task)
    
    
# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(outputdir+'{}_allmetrics.xlsx'.format(output_folder.split('/')[-1]), engine='xlsxwriter')

for metric_name in df_metric_keys:
    print(metric_name)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('will export data to', outputdir)
    
    df_task_list[metric_name] = pd.concat(df_task_list[metric_name], axis=1)

    df_task_copy = df_task_list[metric_name].copy()
    perf_source = df_task_copy.loc['source'].apply(get_mean).mean()
    perf_DANN = df_task_copy.loc['DANN'].apply(get_mean).mean()
    perf_target = df_task_copy.loc['target'].apply(get_mean).mean()
    df_task_list[metric_name].at['source', 'average'] = perf_source
    df_task_list[metric_name].at['target', 'average'] = df_task_copy.loc['target'].apply(get_mean).mean()
    df_task_list[metric_name].at['domain', 'average'] = df_task_copy.loc['domain'].apply(get_mean).mean()
    df_task_list[metric_name]['average'] = df_task_list[metric_name]['average'].astype(object)
    df_task_list[metric_name].at['DANN', 'average'] =  '{:.3f}({:+.3f})'.format(perf_DANN,perf_DANN-perf_source)
    df_task_list[metric_name].at['target', 'average'] =  '{:.3f}({:+.3f})'.format(perf_target,perf_target-perf_source)

    if 'PAD_source' in df_task_copy.index:
        PAD_source = df_task_copy.loc['PAD_source'].apply(get_mean).mean()
        PAD_DANN = df_task_copy.loc['PAD_DANN'].apply(get_mean).mean()
        df_task_list[metric_name].at['PAD_source', 'average'] = PAD_source
        df_task_list[metric_name].at['PAD_DANN', 'average'] = '{:.3f}({:.3f})'.format(PAD_DANN,PAD_DANN-PAD_source)
        df_task_list[metric_name] = df_task_list[metric_name].reindex(['channel_n','batch_size','learning_rate','time_elapsed','num_params',                       'source','DANN','target','domain','PAD_source','PAD_DANN'])  
    else:
        df_task_list[metric_name] = df_task_list[metric_name].reindex(['channel_n','batch_size','learning_rate','time_elapsed','num_params',                   'source','DANN','target','domain'])  
    
    df_task_list[metric_name].loc['time_elapsed'] = df_task_list[metric_name].loc['time_elapsed'].astype(float)
    display(df_task_list[metric_name])
    df_task_list[metric_name].to_csv(outputdir+'df_performance_table_agg_{}.csv'.format(metric_name.split('_')[1]), encoding='utf-8')
    df_task_list[metric_name].to_excel(writer, sheet_name=metric_name.split('_')[1])

writer.save()


# In[13]:


df_task_list[metric_name]


# In[ ]:





# In[ ]:





# In[ ]:


# from matplotlib import animation, rc

# from IPython.display import HTML

# def makeAnimation(raw_data, labels, offset = 0, linesN=3, interval_ms=5, frames_N=100):
#     """makeAnimation makes animation of data and change the background color of the plot based on the labels.

#     Returns anim, you can plot it in ipython notebook using the following two methods:
#     1.  rc('animation', html='html5')
#         anim
#     2.  HTML(anim.to_html5_video())

#     Args:
#         raw_data (numpy array): data vector of accel and HR.
#             raw_data.shape is (number of window, length of each window, number of channels)
#             raw_data[:,:,0] is AccelX (red)
#             raw_data[:,:,1] is AccelY = blue
#             raw_data[:,:,2] is AccelZ = green
#             raw_data[:,:,3] is HR = black
            
#         labels (numpy array): time vector of acc and HR.
#         linesN (scalar): how many lines to display (max=4).
#         interval_ms (scalar): how fast you want to display each frame.
#         frames_N (scalar): how many windows you want to display.

#     Returns:
#         anim: a matplotlib animation object, can be used to display animation on jupyter notebook.
    
#     Example:
#         anim = makeAnimation(raw_data, labels, linesN=4, interval_ms=20, frames_N=1000)
#         rc('animation', html='html5')
#         anim
        
#     Source:
#         http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-notebooks/

#     """
#     # First set up the figure, the axis, and the plot element we want to animate
#     fig, ax = plt.subplots()

#     ax.set_xlim(( 0, 1))
#     # ax.set_ylim((-2, 2))
#     ax.set_ylim((data.min(), data.max()))
    
#     if linesN == 4:
#         axhr_raw = ax.twinx()
#         axhr_raw.set_xlim(( 0, 1))
#         axhr_raw.set_ylim((40, 200))

#     plotlays, plotcols = [linesN], ['red','blue','green','black']
#     lines = []
#     for index in range(linesN):
#         if index == 3:
#             lobj = axhr_raw.plot([],[],lw=2,color=plotcols[index])[0]
#         else:
#             lobj = ax.plot([],[],lw=2,color=plotcols[index])[0]
#         lines.append(lobj)

#     # initialization function: plot the background of each frame
#     def init():
#         for line in lines:
#             line.set_data([],[])
#         return lines    

#     # animation function. This is called sequentially
#     def animate(i):
#         x = np.linspace(0, 1, raw_data.shape[1])

#         for lnum,line in enumerate(lines):
#             line.set_data(x, raw_data[i+offset,:,lnum]) # set data for each line separately. 
#         if labels[i+offset]:
#             ax.set_facecolor((1.0, 0.47, 0.42))
#         else:
#             ax.set_facecolor('white')
#         return lines
    
#     # call the animator. blit=True means only re-draw the parts that have changed.
#     anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                    frames=frames_N, interval=interval_ms, blit=True, repeat=False)
#     return anim


# In[ ]:


# data = src_data.data.detach().cpu().numpy().transpose(0,2,1)
# labels = src_labels.data.detach().cpu().numpy()

# anim = makeAnimation(data, labels, offset=0, linesN=3, interval_ms=200, frames_N=117)
# HTML(anim.to_html5_video())


# In[ ]:


# data.shape, labels.shape
# # data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




