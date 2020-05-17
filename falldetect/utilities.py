import numpy as np

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )

import pandas as pd
pd.set_option('display.max_columns', 500)
from tqdm import tqdm_notebook as tqdm
import os
import sys
from IPython.display import display

import time
import datetime
from datetime import datetime
import json

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

def combine_data(input_dim, dataset_name, sensor_locs, i_CV, inputdir, outputdir):
  axis_n = 3
  data_all = np.zeros((input_dim,axis_n,0))
  labels_all = np.zeros(0)
  i_sub_all = np.zeros(0)

  outputdir_train = outputdir+'/train'
  if not os.path.exists(outputdir_train):
      os.makedirs(outputdir_train)
  print('outputdir for train:', outputdir_train)
  for sensor_loc in sensor_locs:
    inputdir_loc = inputdir+'{}/{}/CV{}/train'.format(dataset_name,sensor_loc,i_CV)
    
    data = data_loader('data', inputdir_loc)
    labels = data_loader('labels', inputdir_loc)
    i_sub = data_loader('i_sub', inputdir_loc)
    # print(data.shape)
    # sys.exit()
    data_all = np.concatenate((data_all,data),axis=2)
    labels_all = np.concatenate((labels_all,labels),axis=0)
    i_sub_all = np.concatenate((i_sub_all,i_sub),axis=0)

  data_saver(data_all, 'data', outputdir_train)
  data_saver(labels_all, 'labels', outputdir_train)
  data_saver(i_sub_all, 'i_sub', outputdir_train)


  data_all = np.zeros((input_dim,axis_n,0))

  labels_all = np.zeros(0)
  i_sub_all = np.zeros(0)
  outputdir_val = outputdir+'/val'
  if not os.path.exists(outputdir_val):
      os.makedirs(outputdir_val)
  print('outputdir for val:', outputdir_val)

  for sensor_loc in sensor_locs:
    inputdir_loc = inputdir+'{}/{}/CV{}/val'.format(dataset_name,sensor_loc,i_CV)
    
    data = data_loader('data', inputdir_loc)
    labels = data_loader('labels', inputdir_loc)
    i_sub = data_loader('i_sub', inputdir_loc)

    data_all = np.concatenate((data_all,data),axis=2)
    labels_all = np.concatenate((labels_all,labels),axis=0)
    i_sub_all = np.concatenate((i_sub_all,i_sub),axis=0)

  data_saver(data_all, 'data', outputdir_val)
  data_saver(labels_all, 'labels', outputdir_val)
  data_saver(i_sub_all, 'i_sub', outputdir_val)

def data_saver(data, name, outputdir):
  """ usage: data_saver(df_merged_interp_alldicts, 'data', outputdir)"""
  outputdir_data = os.path.join(outputdir, name+'.npz')
#   print('outputdir for {}:'.format(name), outputdir_data)
  np.savez(outputdir_data, data=data, allow_pickle=True)
  loaded_data = np.load(outputdir_data, allow_pickle=True)['data']
#     loaded_data = np.load(outputdir_data, allow_pickle=True)['data']
#   print('Are {} save and loadded correctly? '.format(name), np.array_equal(loaded_data, data))
#   print('')
    
def data_loader(name, inputdir):
  """ usage: data = data_loader('data', outputdir)"""
  inputdir_data = os.path.join(inputdir, name+'.npz')
  data = np.load(inputdir_data, allow_pickle=True)['data']
  return data

def export_perofmance(df_performance, CV_n, outputdir):
  df_performance.loc['mean'] = df_performance.iloc[0:CV_n].mean()
  df_performance.loc['std'] = df_performance.iloc[0:CV_n].std()
  print('show df_performance')
  display(df_performance)
  df_performance.to_csv(outputdir+'df_performance.csv')

## export model using torch.save and validate saved model
def export_model(model, loaded_model, outputdir):
  model.eval()
  loaded_model.eval()

  # Save
  torch.save(model.state_dict(), outputdir)
  # Load
  # loaded_model = ConvNet(num_classes=classes_n, input_dim=input_dim).to(device).float()
  # feature_extractor = FeatureExtractor(input_dim=input_dim).to(device).float()
  # class_classifier = ClassClassifier(num_classes=classes_n, input_dim=feature_out_dim).to(device).float()
  # domain_classifier = DomainClassifier(num_classes=2, input_dim=feature_out_dim).to(device).float()
  # loaded_model = CascadedModel(feature_extractor, class_classifier)

  loaded_model.load_state_dict(torch.load(outputdir))

  save_error = 0
  for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
    if p1.data.ne(p2.data).sum() > 0:
      print('model not successfully saved')
      save_error = 1
      break

  if save_error == 0:
    print('model saved successfully')

# perform train_val_split
def train_val_splitter(features_all, labels_all, sub_all, DataNameList_idx_all,
                      i_sub_unique_train, i_sub_unique_val, outputdir):
  data_val = np.zeros((features_all.shape[0],features_all.shape[1],0))
  data_train = np.zeros((features_all.shape[0],features_all.shape[1],0))

  labels_val = np.zeros((0,))
  labels_train = np.zeros((0,))

  i_sub_val = np.zeros((0,))
  i_sub_train = np.zeros((0,))
	
  DataNameList_idx_val = np.zeros((0,))
  DataNameList_idx_train = np.zeros((0,))

  for i_sub in i_sub_unique_train:
      indices_train = np.where(sub_all == i_sub)[0]

      data_train = np.concatenate((data_train, features_all[:,:,indices_train]), axis=2)
      labels_train = np.concatenate((labels_train, labels_all[indices_train,]), axis=0)
      i_sub_train = np.concatenate((i_sub_train, sub_all[indices_train]), axis=0)
      DataNameList_idx_train = np.concatenate((DataNameList_idx_train, DataNameList_idx_all[indices_train]), axis=0)


  for i_sub in i_sub_unique_val:
      indices_val = np.where(sub_all == i_sub)[0]

      data_val = np.concatenate((data_val, features_all[:,:,indices_val]), axis=2)
      labels_val = np.concatenate((labels_val, labels_all[indices_val,]), axis=0)
      i_sub_val = np.concatenate((i_sub_val, sub_all[indices_val]), axis=0)
      DataNameList_idx_val = np.concatenate((DataNameList_idx_val, DataNameList_idx_all[indices_val]), axis=0)

  print('train dimensions:', data_train.shape, labels_train.shape, i_sub_train.shape, DataNameList_idx_train.shape)
  print('val dimensions:', data_val.shape, labels_val.shape, i_sub_val.shape, DataNameList_idx_val.shape)

      
  outputdir_train = os.path.join(outputdir, 'train')
  if not os.path.exists(outputdir_train):
      os.makedirs(outputdir_train)
  print('outputdir for train:', outputdir_train)

  outputdir_val = os.path.join(outputdir, 'val')
  if not os.path.exists(outputdir_val):
      os.makedirs(outputdir_val)
  print('outputdir for val:', outputdir_val)

  data_saver(data_train, 'data', outputdir_train)
  data_saver(labels_train, 'labels', outputdir_train)
  data_saver(i_sub_train, 'i_sub', outputdir_train)
  data_saver(DataNameList_idx_train, 'DataNameList_idx', outputdir_train)

  data_saver(data_val, 'data', outputdir_val)
  data_saver(labels_val, 'labels', outputdir_val)
  data_saver(i_sub_val, 'i_sub', outputdir_val)
  data_saver(DataNameList_idx_val, 'DataNameList_idx', outputdir_val)

  act_all_set = set(labels_train).union(set(labels_val))
  print('All activity ID:', act_all_set)
  if len(set(act_all_set.difference(set(labels_train))))!=0 or len(set(act_all_set.difference(set(labels_val))))!=0:
    print('********* Warning *********')
    print("Missing activity in labels_train:", (act_all_set.difference(set(labels_train)))) 
    print("Missing activity in labels_val:", (act_all_set.difference(set(labels_val)))) 
    print('***************************')

  
  return data_train, data_val, \
         labels_train, labels_val, \
         i_sub_train, i_sub_val, \
         DataNameList_idx_train, DataNameList_idx_val


