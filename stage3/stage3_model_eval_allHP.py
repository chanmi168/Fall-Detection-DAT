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

# In[3]:


parser = argparse.ArgumentParser(description='FD_DAT')
parser.add_argument('--input_folder', metavar='input_folder', help='input_folder',
                    default='../')
parser.add_argument('--output_folder', metavar='output_folder', help='output_folder',
                    default='../')
parser.add_argument('--training_params_file', metavar='training_params_file', help='training_params_file',
                    default='training_params_list.json')
parser.add_argument('--tasks_list', metavar='tasks_list', help='a list of all tasks',
                    default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')
parser.add_argument('--variable_name', metavar='variable_name', help='key in training_params to be displayed on plot',
                    default='HP_name')
# parser.add_argument('--src_names', metavar='src_names', help='a list of src_names',
#                     default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')
# parser.add_argument('--tgt_names', metavar='tgt_names', help='a list of tgt_names',
#                     default='UMAFall_waist_UPFall_belt UPFall_wrist_UMAFall_ankle')

# checklist 2: comment first line, uncomment second line seizures_FN

# args = parser.parse_args(['--input_folder', '../../data_mic/stage2/modeloutput_18hz_5fold_UPFall_UMAFall_cross-config_diffCV',
#                           '--output_folder', '../../data_mic/stage3/test',


# args = parser.parse_args(['--input_folder', '../../data_mic/stage2/modeloutput_WithoutNormal_18hz_5fold_UPFall_UMAFall_cross-config_diffCV',
# args = parser.parse_args(['--input_folder', '../../data_mic/stage2/modeloutput_18hz_5fold_UPFall_UMAFall_cross-config_diffCV_weighted_refactor',
#                           '--output_folder', '../../data_mic/stage3/test',
#                           '--training_params_file', 'training_params_list_fixed.json',
# #                           '--tasks_list', 'UMAFall_chest-UPFall_neck UMAFall_wrist-UPFall_wrist UMAFall_waist-UPFall_belt UMAFall_leg-UPFall_rightpocket UMAFall_ankle-UPFall_ankle',
#                           '--tasks_list', 'UPFall_neck-UMAFall_chest UPFall_wrist-UMAFall_wrist UPFall_belt-UMAFall_waist UPFall_rightpocket-UMAFall_leg UPFall_ankle-UMAFall_ankle',
#                           '--variable_name', 'HP_name',])

#                           '--src_names', 'UPFall_neck UPFall_wrist UPFall_belt UPFall_rightpocket UPFall_ankle',
#                           '--tgt_names', 'UMAFall_chest UMAFall_wrist UMAFall_waist UMAFall_leg UMAFall_ankle'])

args = parser.parse_args()


# In[ ]:





# In[4]:


home_dir = home+'/project_FDDAT/'
input_folder = args.input_folder
output_folder = args.output_folder
training_params_file = args.training_params_file

tasks_list = []
for item in args.tasks_list.split(' '):
    tasks_list.append((item.split('-')[0], item.split('-')[1]))
    
# src_domains = args.src_names.split(' ')
# tgt_domains = args.tgt_names.split(' ')
variable_name = args.variable_name

inputdir = input_folder + '/'
outputdir = output_folder + '/'
if not os.path.exists(outputdir):
    os.makedirs(outputdir)


# In[5]:


outputdir


# In[6]:


with open('../stage2/'+training_params_file) as json_file:
    training_params_list = json.load(json_file)

# TODO: need to fix once training_params_list is fixed
# training_params_list.pop(-2)
training_params_list


# In[ ]:





# In[7]:



# def plot_task_metric(df_temp, metric_name, outputdir):
#     source_means = df_temp.loc['source',df_temp.columns != 'average'].apply(get_mean).values
#     DANN_means = df_temp.loc['DANN',df_temp.columns != 'average'].apply(get_mean).values
#     task_names = df_temp.columns[df_temp.columns != 'average']

#     fig = plt.figure(figsize=(5, 5), dpi=dpi)
#     ax = fig.add_subplot(1, 1, 1)
#     ax.scatter(source_means, DANN_means, s=40, marker='o')
#     ax_xlim = ax.get_xlim()
#     ax_ylim = ax.get_ylim()
#     ax.plot([0, 1], [0, 1], c=".3", linewidth=1, alpha=0.4)
#     ax.set_title('{}'.format(metric_name.split('_')[1]), fontsize=20)
#     ax.set_xlabel('source_means', fontsize=15)
#     ax.set_ylabel('DANN_means', fontsize=15)   # relative to plt.rcParams['font.size']
#     ax.set_xlim(min(ax_xlim[0], ax_ylim[0]), max(ax_xlim[1], ax_ylim[1]))
#     ax.set_ylim(min(ax_xlim[0], ax_ylim[0]), max(ax_xlim[1], ax_ylim[1]))

#     for i, txt in enumerate(task_names):
#         ax.annotate(txt, (source_means[i], DANN_means[i]), fontsize=10, textcoords="offset points", xytext=(0,10), ha='center')
#     fig.savefig(outputdir+'scatter_{}.png'.format(metric_name.split('_')[1]))


# In[ ]:





# In[8]:


df_metric_keys = ['df_acc', 'df_sensitivity', 'df_precision', 'df_F1']
metric_names = ['acc', 'sensitivity', 'precision', 'F1','PAD']
training_setting_list = ['source', 'DANN', 'target', 'improvement']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


dict_task_name_list = []
for training_params in training_params_list:
#     dict_task_name_list.append('N_ch={}'.format(training_params['channel_n']))
    dict_task_name_list.append(training_params['HP_name'])

dict_task_name_list


# In[10]:


print('HP_name\t\tchannel_n')
for training_params in training_params_list:
    print('{}\t\t{}'.format(training_params['HP_name'], training_params['channel_n']))


# In[13]:


df_task_list = dict( zip(df_metric_keys,[[], [], [], []]))
training_types = ['source','dann','target']

dict_task_all = {}
for task_item in tasks_list:
    (src_name, tgt_name) = task_item
    print(task_item)
    
    dict_task = dict( zip( dict_task_name_list,[{},{},{},{},{},{}] ) )

    training_type = 'source'
    for training_params in training_params_list:
        df_list = []
        for i_rep in range(training_params['rep_n']):
            df_inputdir = inputdir+src_name+'_'+tgt_name+'/{}/{}/rep{}/df_performance.csv'.format(training_params['HP_name'],training_type,i_rep)
            df = pd.read_csv(df_inputdir, index_col=0).iloc[0:training_params['CV_n']][['val_tgt_acc','val_tgt_sensitivity','val_tgt_precision','val_tgt_F1','PAD']]
            df = df.rename(columns={'val_tgt_acc':'acc','val_tgt_sensitivity':'sensitivity','val_tgt_precision':'precision','val_tgt_F1':'F1'})
            df_list.append(df)
        
#         dict_task['N_ch={}'.format(training_params['channel_n'])]['performance_{}'.format(training_type)] = pd.concat(df_list,ignore_index=True)
        dict_task[training_params['HP_name']]['performance_{}'.format(training_type)] = pd.concat(df_list,ignore_index=True)

    training_type = 'dann'
    for training_params in training_params_list:
        df_list = []
        for i_rep in range(training_params['rep_n']):
            df_inputdir = inputdir+src_name+'_'+tgt_name+'/{}/{}/rep{}/df_performance.csv'.format(training_params['HP_name'],training_type,i_rep)
            df = pd.read_csv(df_inputdir, index_col=0).iloc[0:training_params['CV_n']][['val_tgt_acc','val_tgt_sensitivity','val_tgt_precision','val_tgt_F1','PAD']]
#             df = df.rename(columns={'val_tgt_class_acc':'acc','val_tgt_class_sensitivity':'sensitivity','val_tgt_class_precision':'precision','val_tgt_class_F1':'F1'})
            df = df.rename(columns={'val_tgt_acc':'acc','val_tgt_sensitivity':'sensitivity','val_tgt_precision':'precision','val_tgt_F1':'F1'})
            df_list.append(df)

#         dict_task['N_ch={}'.format(training_params['channel_n'])]['performance_{}'.format(training_type)] = pd.concat(df_list,ignore_index=True)
        dict_task[training_params['HP_name']]['performance_{}'.format(training_type)] = pd.concat(df_list,ignore_index=True)


    training_type = 'target'
    for training_params in training_params_list:
        df_list = []
        for i_rep in range(training_params['rep_n']):
            df_inputdir = inputdir+src_name+'_'+tgt_name+'/{}/{}/rep{}/df_performance.csv'.format(training_params['HP_name'],training_type,i_rep)
            df = pd.read_csv(df_inputdir, index_col=0).iloc[0:training_params['CV_n']][['val_src_acc','val_src_sensitivity','val_src_precision','val_src_F1','PAD']]
            df = df.rename(columns={'val_src_acc':'acc','val_src_sensitivity':'sensitivity','val_src_precision':'precision','val_src_F1':'F1'})
            df_list.append(df)

#         dict_task['N_ch={}'.format(training_params['channel_n'])]['performance_{}'.format(training_type)] = pd.concat(df_list,ignore_index=True)
        dict_task[training_params['HP_name']]['performance_{}'.format(training_type)] = pd.concat(df_list,ignore_index=True)

    dict_task_all[src_name+'_'+tgt_name] = dict_task        


# In[ ]:





# In[ ]:





# In[14]:


color_dict = {'Green': '#3cb44b', 
              'Sunglow': '#FFD43B',
              'Red': '#e6194b', 
              'Blue': '#0082c8', 

              'Teal': '#008080', 

              'Maroon': '#800000', 
              'Navy': '#000080', 
              'Mint': '#aaffc3', 
              'Yellow': '#ffe119', 

              'Orange': '#f58231', 
              'Purple': '#911eb4', 
              'Cyan': '#46f0f0', 
              'Magenta': '#e6194b', 
              'Lime': '#d2f53c', 
              'Pink': '#fabebe', 
              'Lavender': '#e6beff', 

              'Brown': '#aa6e28', 
              'Mint': '#aaffc3', 
              'Olive': '#808000', 
              'Coral': '#ffd8b1',  
              'Grey': '#808080', 
#               'Lavender': '#e6beff', 
             }
colornames = list(color_dict.keys())


# In[ ]:





# In[ ]:





# In[15]:


# def plot_metrics(task_name, metric_names, HP_name, dict_task_HP, outputdir):
def plot_metrics(task_name, metric_names, key, dict_task, variable_name, training_params_list, outputdir):
    fontsize_label = {
        'subtitle': 20,
        'axtitle': 17,
        'xytitle': 17,
        'annotate':6,
    }
    
    dann_color = 'Blue'
    tgt_color = 'Green'
    
    dict_task_HP = dict_task[key]
    fig = plt.figure(figsize=(len(metric_names)*5, 5), dpi=100+len(metric_names)*5)
#     fig.suptitle('{} ({})'.format(task_name,key), fontsize=23, y=1.06)

    channel_n = next(training_params for training_params in training_params_list if training_params['HP_name'] == key)[variable_name]
    fig.suptitle('{}\n({}={})'.format(task_name,variable_name,channel_n), fontsize=fontsize_label['subtitle'], y=1.12)


    for i, metric_name in enumerate(metric_names):
        source_dpt = dict_task_HP['performance_source'][metric_name].values
        dann_dpt = dict_task_HP['performance_dann'][metric_name].values
        target_dpt = dict_task_HP['performance_target'][metric_name].values

#         ax = fig.add_subplot(2, len(metric_names), i+1)
        ax = fig.add_subplot(1, len(metric_names), i+1)
        ax.scatter(source_dpt, dann_dpt, s=40, marker='o', alpha=0.5, c=color_dict[dann_color])
#         ax.set_title('{}({:+.4f})'.format(metric_name,np.nanmean(dann_dpt)-np.nanmean(source_dpt)), fontsize=fontsize_label['axtitle'])
        ax.set_xlabel('source({:.4f}±{:.4f})'.format(np.nanmean(source_dpt),np.nanstd(source_dpt)), fontsize=fontsize_label['xytitle'])
        ax.set_ylabel('DANN({:.4f}±{:.4f})'.format(np.nanmean(dann_dpt),np.nanstd(dann_dpt)), fontsize=fontsize_label['xytitle'], c=color_dict[dann_color])

        ax_r = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax_r.scatter(source_dpt, target_dpt, s=40, marker='o', alpha=0.5, c=color_dict[tgt_color])
        ax_r.set_ylabel('target({:.4f}±{:.4f})'.format(np.nanmean(target_dpt),np.nanstd(target_dpt)), fontsize=fontsize_label['xytitle'], c=color_dict[tgt_color]) 

        if metric_name == 'PAD':
            ax_xlim = ax.get_xlim()
            ax_ylim = ax.get_ylim()
            ax_r_xlim = ax_r.get_xlim()
            ax_r_ylim = ax_r.get_ylim()
            x_min = min(ax_xlim+ax_r_xlim)
            x_max = max(ax_xlim+ax_r_xlim)
            y_min = min(ax_ylim+ax_r_ylim)
            y_max = max(ax_ylim+ax_r_ylim)
            ax.plot([x_min, x_max], [y_min, y_max], c=".3", linewidth=1, alpha=0.4)
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            ax_r.set_ylim(y_min,y_max)

        else:
            ax.plot([0, 1], [0, 1], c=".3", linewidth=1, alpha=0.4)
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax_r.set_ylim(0,1)
        
        for i in range(source_dpt.shape[0]):
            ax.annotate(i, (source_dpt[i], dann_dpt[i]),alpha=0.9,fontsize=fontsize_label['annotate'], c=color_dict[dann_color])
            ax_r.annotate(i, (source_dpt[i], target_dpt[i]),alpha=0.9,fontsize=fontsize_label['annotate'], c=color_dict[tgt_color])

        ax.set_title('{}({:+.4f} / {:+.4f})'.format(metric_name,np.nanmean(dann_dpt)-np.nanmean(source_dpt),np.nanmean(target_dpt)-np.nanmean(source_dpt)), fontsize=fontsize_label['axtitle'])

#     for i, metric_name in enumerate(metric_names):
#         source_dpt = dict_task_HP['performance_source'][metric_name].values
#         dann_dpt = dict_task_HP['performance_dann'][metric_name].values
#         target_dpt = dict_task_HP['performance_target'][metric_name].values

#         ax = fig.add_subplot(2, len(metric_names), i+len(metric_names)+1)
#         ax.scatter(source_dpt, target_dpt, s=40, marker='o', alpha=0.6)
#         ax_xlim = ax.get_xlim()
#         ax_ylim = ax.get_ylim()
#         ax.plot([0, 1], [0, 1], c=".3", linewidth=1, alpha=0.4)
#         ax.set_title('{}({:+.4f})'.format(metric_name,np.nanmean(target_dpt)-np.nanmean(source_dpt)), fontsize=fontsize_label['axtitle'])
#         ax.set_xlabel('source({:.4f}±{:.4f})'.format(np.nanmean(source_dpt),np.nanstd(source_dpt)), fontsize=fontsize_label['xytitle'])
#         ax.set_ylabel('target({:.4f}±{:.4f})'.format(np.nanmean(target_dpt),np.nanstd(target_dpt)), fontsize=fontsize_label['xytitle']) 

# #         ax.set_xlim(min(ax_xlim[0], ax_ylim[0]), max(ax_xlim[1], ax_ylim[1]))
# #         ax.set_ylim(min(ax_xlim[0], ax_ylim[0]), max(ax_xlim[1], ax_ylim[1]))
#         ax.set_xlim(0,1)
#         ax.set_ylim(0,1)
        
#         for i in range(source_dpt.shape[0]):
#             ax.annotate(i, (source_dpt[i], target_dpt[i]),alpha=0.7,fontsize=fontsize_label['annotate'])
        
    fig.tight_layout()
    fig_dir = outputdir+task_name
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.savefig(fig_dir+'/scatter_{}.png'.format(key), bbox_inches = "tight")


for task_name, dict_task in dict_task_all.items():
    for key in dict_task.keys():
        plot_metrics(task_name, ['F1'], key, dict_task, variable_name, training_params_list, outputdir)
#         sys.exit()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:


def F1_checker_each(dict_task_HP):
    for key in dict_task_HP.keys():
        arr_sens = dict_task_HP[key]['sensitivity'].values
        arr_prec = dict_task_HP[key]['precision'].values
        arr_F1 = dict_task_HP[key]['F1'].values

        test_F1 = 2*(arr_sens*arr_prec)/(arr_sens+arr_prec)

        if np.nansum(np.abs(test_F1-arr_F1)) > 0.00001:
            print('***      F1 for a CV, a rep computed incorrectly in {}     ***'.format(key))


# In[17]:


def F1_checker_mean(dict_df):
    for column_name in dict_df['sensitivity'].columns.values:
        for row_name in dict_df['sensitivity'].index.values:
            float_sens = float(dict_df['sensitivity'].loc[row_name, column_name].split('±')[0])
            float_prec = float(dict_df['precision'].loc[row_name, column_name].split('±')[0])
            float_F1 = float(dict_df['F1'].loc[row_name, column_name].split('±')[0])

            test_F1 = 2*(float_sens*float_prec)/(float_sens+float_prec)

            if np.abs(float_F1-test_F1) > 0.00001:
                print('***      mean F1 for all CV, all rep computed incorrectly in {}     ***'.format(key))
                # # print('float_sens:', float_sens)
                # # print('float_prec:', float_prec)
                # # print('float_F1:', float_F1)
                # # print('test_F1:', test_F1)


# In[ ]:





# In[18]:


debug_F1 = False

dict_df_all = {}


for task_name, dict_task in dict_task_all.items():
    df = pd.DataFrame(columns=list(dict_task.keys()),index=['source','DANN','target'])
    dict_df = dict( zip(metric_names,[df.copy(), df.copy(), df.copy(), df.copy(), df.copy()]))

    for key in dict_task.keys():
        dict_task_HP = dict_task[key]
        
        if debug_F1:
            F1_checker(dict_task_HP)

        for i, metric_name in enumerate(metric_names):
            source_dpt = dict_task_HP['performance_source'][metric_name].values
            dann_dpt = dict_task_HP['performance_dann'][metric_name].values
            target_dpt = dict_task_HP['performance_target'][metric_name].values
            
            dict_df[metric_name].loc['source', key] = '{:.4f}±{:.4f}'.format(np.nanmean(source_dpt),np.nanstd(source_dpt))
            dict_df[metric_name].loc['DANN', key] = '{:.4f}±{:.4f}'.format(np.nanmean(dann_dpt),np.nanstd(dann_dpt))
            dict_df[metric_name].loc['target', key] = '{:.4f}±{:.4f}'.format(np.nanmean(target_dpt),np.nanstd(target_dpt))

    # Create a Pandas Excel writer using XlsxWriter as the engine
    df_outputdir = outputdir+task_name
    if not os.path.exists(df_outputdir):
        os.makedirs(df_outputdir)
        
    writer = pd.ExcelWriter(df_outputdir+'/allmetrics.xlsx', engine='xlsxwriter')

    for i, metric_name in enumerate(metric_names):
        print(task_name, metric_name)
        display(dict_df[metric_name])
        dict_df[metric_name].to_excel(writer, sheet_name=metric_name)
        
        if debug_F1:
            F1_checker_mean(dict_df)

    writer.save()

    dict_df_all[task_name] = dict_df.copy()
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




