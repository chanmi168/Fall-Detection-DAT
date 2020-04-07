#!/usr/bin/env python
# coding: utf-8

# # import packages and get authenticated

# In[1]:


# from google.colab import driveA
# drive.mount('drive')


# In[2]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 500)

# Plotting
# checklist 1: comment inline, uncomment Agg
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )

# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
import argparse
import os
import sys
sys.path.append('/content/drive/My Drive/中研院/repo/')
sys.path.append('~/project_FDDAT/repo/')
sys.path.append('../') # add this line so Data and data are visible in this file

from falldetect.utilities import *

import time
import datetime
from datetime import datetime

from sklearn.decomposition import PCA

from os.path import expanduser
home = expanduser("~")
# home_dir = home+'/project_FDDAT/'

# split_mode = 'LOO'
# split_mode = '5fold'


# # Get user inputs
# In ipython notebook, these are hardcoded. In production python code, use parsers to provide these inputs

# In[3]:


parser = argparse.ArgumentParser(description='FD_DAT')
parser.add_argument('--dataset_name', metavar='dataset_name', help='dataset_name',
                    default='UMAFall')
parser.add_argument('--sensor_loc', metavar='sensor_loc', help='sensor_loc',
                    default='ankle')
parser.add_argument('--outputdir', metavar='outputdir', help='path to outputdir',
                    default='../')
# parser.add_argument('--home_dir', metavar='home_dir', help='path to home_dir',
#                     default='../')
parser.add_argument('--split_mode', metavar='split_mode', help='split_mode',
                    default='5fold')
# split_mode = 'LOO'
# split_mode = '5fold'

# checklist 2: comment first line, uncomment second line seizures_FN
# args = parser.parse_args(['--dataset_name', 'UMAFall', 
#                           '--sensor_loc', 'ankle',
# #                           '--home_dir', home+'/project_FDDAT/',
#                           '--split_mode', '5fold',])
                          


args = parser.parse_args()


# In[4]:


dataset_name = args.dataset_name
sensor_loc = args.sensor_loc
home_dir = home+'/project_FDDAT/'
split_mode = args.split_mode


# # load data_management (all) first

# In[5]:


# dataset_name = 'UMAFall'
# # sensor_loc = 'waist'
# # sensor_loc = 'wrist'
# # sensor_loc = 'leg'
# # sensor_loc = 'chest'
# sensor_loc = 'ankle'

# home_dir = 


# resampled, 18hz
# DataNameList_inputdir = '/content/drive/My Drive/中研院/Data/{}/ImpactWindow_Resample/18hz/{}/IP_{}_DataNameList_{}.csv'.format(dataset_name, sensor_loc, dataset_name, sensor_loc)
# impact_inputdir = '/content/drive/My Drive/中研院/Data/{}/ImpactWindow_Resample/18hz/{}/'.format(dataset_name, sensor_loc)
# outputdir = '/content/drive/My Drive/中研院/data_mic/stage1_preprocessed_18hz_LOO/{}/{}/'.format(dataset_name, sensor_loc)

DataNameList_inputdir = home_dir + 'Data/{}/ImpactWindow_Resample/18hz/{}/IP_{}_DataNameList_{}.csv'.format(dataset_name, sensor_loc, dataset_name, sensor_loc)
impact_inputdir = home_dir + 'Data/{}/ImpactWindow_Resample/18hz/{}/'.format(dataset_name, sensor_loc)
outputdir = home_dir + 'data_mic/stage1_preprocessed_18hz_{}/{}/{}/'.format(split_mode, dataset_name, sensor_loc)
print('will export data to', outputdir)

df = pd.read_csv(DataNameList_inputdir)


# In[6]:


act_names = df['Activity_ID'].unique()
act_embeddings = { act_names[i] : i for i in range(0, act_names.shape[0] ) }
act_embeddings


# In[ ]:





# In[7]:


temp = pd.read_csv(impact_inputdir+df['x_DataName'][0], header=None)

window_length = temp.shape[0]
samples_n = df.shape[0]

data_all = np.zeros((window_length,3,samples_n))
actlabels_all = np.zeros((samples_n,))
sub_all = np.zeros((samples_n,))

i = 0
for filename in tqdm(df['x_DataName']):
    sub_id = int(filename.split('_')[0])
    activity_id = act_embeddings[filename.split('_')[1]]
    location = filename.split('_')[3][:-4]

    df_imp = pd.read_csv(impact_inputdir+filename, header=None)

    data_all[:,:,i] = df_imp.to_numpy()
    actlabels_all[i] = activity_id
    sub_all[i] = sub_id
    i += 1


# In[8]:


fall_n = (actlabels_all==10).sum()+(actlabels_all==11).sum()+(actlabels_all==12).sum()
adl_n = data_all.shape[2]-fall_n
print('fall_n, adl_n:', fall_n, adl_n)


# In[9]:


np.where(actlabels_all==10)[0]


# In[ ]:





# In[10]:


samples_n = data_all.shape[2]
labels_n = np.shape(np.unique(actlabels_all))[0]
subjects_n = np.shape(np.unique(sub_all))[0]
print('finished reading data in data_management {} at {}'.format(dataset_name, sensor_loc))
print('Dimension of data', data_all.shape)
print('number of activities', labels_n)
print('number of subject', subjects_n)

if split_mode == 'LOO':
    CV_n = subjects_n
elif split_mode == '5fold':
    CV_n = int(split_mode.split('fold')[0])
    
print('will split data into {} folds'.format(CV_n))


# In[11]:


sub_all.shape


# In[12]:


count = 0

for i in range(data_all.shape[2]):
    if actlabels_all[i] < 10:
        continue
    plt.plot(data_all[:,0,i], label='x')
    plt.plot(data_all[:,1,i], label='y')
    plt.plot(data_all[:,2,i], label='z')
    plt.ylabel('acc value (a.u.)')
    plt.xlabel('time (sample)')
    plt.legend(loc='upper right')
    if actlabels_all[i] in [10,11,12]:
        plt.title('subject {}, act {} -Fall-'.format(sub_all[i], actlabels_all[i]))
    else:
        plt.title('subject {}, act {} -ADL-'.format(sub_all[i], actlabels_all[i]))
    plt.show()

    count += 1
    if count == 10:
        break


# In[ ]:





# In[13]:


# ughhhhh  class 4, 9, 13, 14 are underrepresented
plt.hist(actlabels_all, bins=np.arange(labels_n+1)-0.5, alpha=0.5, histtype='bar', ec='black')
plt.xlabel('activity label')
plt.ylabel('sample N')
plt.title('activity histogram for all data')


# # split data into train and val (1:1)
# split by sample_id

# In[14]:


i_sub_unique_all = np.unique(sub_all)
i_sub_excluded = []
for i_sub in i_sub_unique_all:
#     print(i_sub)
    idx_sub = np.where(sub_all==i_sub)[0]
    idx_sub_fall = np.where((actlabels_all[idx_sub]==10) | (actlabels_all[idx_sub]==11) | (actlabels_all[idx_sub]==12))[0]
    if len(idx_sub_fall)==0:
#         print('i_sub {} has no fall data, will exclude'.format(int(i_sub)))
        i_sub_excluded.append(int(i_sub))

print('i_sub {} has no fall data, will exclude'.format(i_sub_excluded))

i_sub_unique = np.array(list(set(i_sub_unique_all) - set(i_sub_excluded)))
print(i_sub_unique_all)
print(i_sub_excluded)
print(i_sub_unique)


# In[15]:


from sklearn.model_selection import KFold
# kfold = 5
kfold = CV_n
kf = KFold(n_splits=kfold, random_state=0, shuffle=False)

# i_sub_unique = np.unique(sub_all)
print('all i_sub_unique', i_sub_unique)
np.random.seed(0)
np.random.shuffle(i_sub_unique)
kf.get_n_splits(i_sub_unique)
print(kf)  

for train_index, val_index in kf.split(i_sub_unique):
    print("TRAIN:", i_sub_unique[train_index], "VAL:", i_sub_unique[val_index])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Split based on CV results then 

# In[ ]:





# In[16]:


i_CV = 0

for train_idx, val_idx in kf.split(i_sub_unique):
    print("Sub ID | TRAIN:", i_sub_unique[train_idx], "VAL:", i_sub_unique[val_idx])

    train_val_splitter(data_all, actlabels_all, sub_all, 
                       i_sub_unique[train_idx], i_sub_unique[val_idx], outputdir+'CV'+str(i_CV))

    i_CV = i_CV + 1


# In[ ]:





# In[ ]:




