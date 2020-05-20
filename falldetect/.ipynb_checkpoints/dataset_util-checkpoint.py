import numpy as np

import pandas as pd
pd.set_option('display.max_columns', 500)
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
import os
import sys
sys.path.append('/content/drive/My Drive/中研院/repo/')

from falldetect.utilities import *
from falldetect.dataset_util import *

import time
import datetime
from datetime import datetime
import json

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

val_batch_size = 5000
	
class FallDataset(Dataset):
  def __init__(self, data, labels, DataNameList_idx):
      self.data = torch.FloatTensor(data)
      self.labels = torch.FloatTensor(labels)
      self.DataNameList_idx = DataNameList_idx
      # self.data = torch.LongTensor(data)
      # self.labels = torch.LongTensor(labels)

  def __getitem__(self, index):
      x = self.data[index,:,:]
      y = self.labels[index]
      return x, y
#       return index, x, y

  def __len__(self):
      return len(self.data)
	
	
def get_data_loader(inputdir, i_CV, batch_size, learning_rate, training_params):
  print('Working on get_data_loader for {}...'.format(inputdir.split('/')[5:7]))

  train_inputdir = inputdir+'CV{}/train'.format(i_CV)
  val_inputdir = inputdir+'CV{}/val'.format(i_CV)

  train_data = data_loader('data', train_inputdir).transpose(2,1,0)[:,:,0:66]
  val_data = data_loader('data', val_inputdir).transpose(2,1,0)[:,:,0:66]

  train_labels = data_loader('labels', train_inputdir)
  val_labels = data_loader('labels', val_inputdir)

  train_i_sub = data_loader('i_sub', train_inputdir)
  val_i_sub = data_loader('i_sub', val_inputdir)

  train_DataNameList_idx = data_loader('DataNameList_idx', train_inputdir)
  val_DataNameList_idx = data_loader('DataNameList_idx', val_inputdir)

  print('train_data shape: {} [{}:{}]'.format(train_data.shape, (train_labels==0).sum(), (train_labels==1).sum()))
  print('val_data shape: {} [{}:{}]'.format(val_data.shape, (val_labels==0).sum(), (val_labels==1).sum()))

  train_size = train_labels.shape[0]
  val_size = val_labels.shape[0]
  input_dim = train_data.shape[2]

  # convert labels from multi-class activities to binary (fall/ADL)
  train_labels_binary = (train_labels==1).astype(int)
  val_labels_binary = (val_labels==1).astype(int)

  train_dataset = FallDataset(train_data, train_labels_binary, train_DataNameList_idx)
  val_dataset = FallDataset(val_data, val_labels_binary, val_DataNameList_idx)

#   use_WeightedRandomSampler = True
# data loader (with sampler)
  if training_params['use_WeightedRandomSampler']:
# 	  print((train_labels_binary==0).sum(), (train_labels_binary==1).sum(), train_labels_binary.shape)
	  class_sample_counts = [(train_labels_binary==0).sum(), (train_labels_binary==1).sum()]
	  k = class_sample_counts[0]*class_sample_counts[1]/(class_sample_counts[0]+class_sample_counts[1])
	  weights = 1. / torch.tensor(class_sample_counts, dtype=torch.float) * k
	  print('counts for labels', class_sample_counts)
	  print('weights for labels', weights)
	  samples_weights = weights[train_labels_binary]
	  train_sampler =  torch.utils.data.sampler.WeightedRandomSampler(
		  weights=samples_weights,
		  num_samples=len(samples_weights),
		  replacement=True)

	  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
												batch_size=batch_size, 
												sampler=train_sampler)

	  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
												batch_size=val_batch_size, 
												shuffle=False)
# data loader (no sampler)

  else:
	  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)
	  val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=val_batch_size, 
                                            shuffle=False)

  return train_loader, val_loader


# # TBD
# def get_UMAFall_loader(inputdir, i_CV, batch_size, learning_rate):
#   print('Working on get_UMAFall_loader...')

# #   val_batch_size = 5000
#   # val_batch_size = batch_size

#   train_inputdir = inputdir+'CV{}/train'.format(i_CV)
#   val_inputdir = inputdir+'CV{}/val'.format(i_CV)

#   train_data = data_loader('data', train_inputdir).transpose(2,1,0)[:,:,0:66]
#   val_data = data_loader('data', val_inputdir).transpose(2,1,0)[:,:,0:66]
#   # train_data = data_loader('data', train_inputdir).transpose(2,1,0)
#   # val_data = data_loader('data', val_inputdir).transpose(2,1,0)

#   train_labels = data_loader('labels', train_inputdir)
#   val_labels = data_loader('labels', val_inputdir)

#   train_i_sub = data_loader('i_sub', train_inputdir)
#   val_i_sub = data_loader('i_sub', val_inputdir)

#   print('train_data shape:', train_data.shape)
#   print('val_data shape:', val_data.shape)

#   train_size = train_labels.shape[0]
#   val_size = val_labels.shape[0]
#   input_dim = train_data.shape[2]

#   # convert labels from multi-class activities to binary (fall/ADL)
#   train_labels_binary = ((train_labels==10)|(train_labels==11)|(train_labels==12)).astype(int)
#   val_labels_binary = ((val_labels==10)|(val_labels==11)|(val_labels==12)).astype(int)

#   train_dataset = FallDataset(train_data, train_labels_binary)
#   val_dataset = FallDataset(val_data, val_labels_binary)
#   # data loader
#   train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                             batch_size=batch_size, 
#                                             shuffle=True)
#   val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                             batch_size=val_batch_size, 
#                                             shuffle=False)

#   return train_loader, val_loader

# def get_UPFall_loader(inputdir, i_CV, batch_size, learning_rate):
#   print('Working on get_UPFall_loader...')

# #   val_batch_size = 5000
#   # val_batch_size = batch_size

#   train_inputdir = inputdir+'CV{}/train'.format(i_CV)
#   val_inputdir = inputdir+'CV{}/val'.format(i_CV)

#   # train_data = data_loader('data', train_inputdir).transpose(2,1,0)
#   # val_data = data_loader('data', val_inputdir).transpose(2,1,0)
#   train_data = data_loader('data', train_inputdir).transpose(2,1,0)[:,:,0:66]
#   val_data = data_loader('data', val_inputdir).transpose(2,1,0)[:,:,0:66]

#   train_labels = data_loader('labels', train_inputdir)
#   val_labels = data_loader('labels', val_inputdir)

#   train_i_sub = data_loader('i_sub', train_inputdir)
#   val_i_sub = data_loader('i_sub', val_inputdir)

#   print('train_data shape:', train_data.shape)
#   print('val_data shape:', val_data.shape)

#   train_size = train_labels.shape[0]
#   val_size = val_labels.shape[0]
#   input_dim = train_data.shape[2]

#   # convert labels from multi-class activities to binary (fall/ADL)
#   train_labels_binary = ((train_labels==1)|(train_labels==2)|(train_labels==3)|(train_labels==4)|(train_labels==5)).astype(int)
#   val_labels_binary = ((val_labels==1)|(val_labels==2)|(val_labels==3)|(val_labels==4)|(val_labels==5)).astype(int)

#   train_dataset = FallDataset(train_data, train_labels_binary)
#   val_dataset = FallDataset(val_data, val_labels_binary)
#   # data loader
#   train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                             batch_size=batch_size, 
#                                             shuffle=True)
#   val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                             batch_size=val_batch_size, 
#                                             shuffle=False)

#   return train_loader, val_loader

# def get_SFDLA_loader(inputdir, i_CV, batch_size, learning_rate):
#   print('Working on get_SFDLA_loader...')

# #   val_batch_size = 5000
#   # val_batch_size = batch_size

#   train_inputdir = inputdir+'CV{}/train'.format(i_CV)
#   val_inputdir = inputdir+'CV{}/val'.format(i_CV)

#   # train_data = data_loader('data', train_inputdir).transpose(2,1,0)
#   # val_data = data_loader('data', val_inputdir).transpose(2,1,0)
#   train_data = data_loader('data', train_inputdir).transpose(2,1,0)[:,:,0:66]
#   val_data = data_loader('data', val_inputdir).transpose(2,1,0)[:,:,0:66]

#   train_labels = data_loader('labels', train_inputdir)
#   val_labels = data_loader('labels', val_inputdir)

#   train_i_sub = data_loader('i_sub', train_inputdir)
#   val_i_sub = data_loader('i_sub', val_inputdir)

#   print('train_data shape:', train_data.shape)
#   print('val_data shape:', val_data.shape)

#   train_size = train_labels.shape[0]
#   val_size = val_labels.shape[0]
#   input_dim = train_data.shape[2]

#   # convert labels from multi-class activities to binary (fall/ADL)
#   train_labels_binary = (train_labels>=900).astype(int)
#   val_labels_binary = (val_labels>=900).astype(int)

#   train_dataset = FallDataset(train_data, train_labels_binary)
#   val_dataset = FallDataset(val_data, val_labels_binary)
#   # data loader
#   train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                             batch_size=batch_size, 
#                                             shuffle=True)
#   val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
#                                             batch_size=val_batch_size, 
#                                             shuffle=False)

#   return train_loader, val_loader
