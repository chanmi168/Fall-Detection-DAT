import numpy as np

import pandas as pd
pd.set_option('display.max_columns', 500)
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
import os
import math
import sys
sys.path.append('/content/drive/My Drive/中研院/repo/')

from falldetect.utilities import *
from falldetect.models import *
from falldetect.dataset_util import *
from falldetect.eval_util import *

import time
import datetime
from datetime import datetime
import json
import copy 

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

λ = 1

# def train_epoch_dann(src_loader, tgt_loader, device, dann, class_criterion, domain_criterion, optimizer, epoch, training_mode):
#   dann.train()

#   total_src_class_loss = 0
#   total_tgt_class_loss = 0
#   total_src_domain_loss = 0
#   total_tgt_domain_loss = 0

#   domain_TPTN = 0
	
#   src_TP = 0
#   src_FP = 0
#   src_TN = 0
#   src_FN = 0
	
#   tgt_TP = 0
#   tgt_FP = 0
#   tgt_TN = 0
#   tgt_FN = 0

#   for i, (sdata, tdata) in enumerate(zip(src_loader, tgt_loader)):
#     src_data, src_labels = sdata
#     tgt_data, tgt_labels = tdata

#     src_data = src_data.to(device)
#     src_labels = src_labels.to(device).long()
#     tgt_data = tgt_data.to(device)
#     tgt_labels = tgt_labels.to(device).long()

#     # prepare domain labels
#     src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
#     tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

#     src_feature, src_class_out, src_domain_out = dann(src_data)
#     tgt_feature, tgt_class_out, tgt_domain_out = dann(tgt_data)

#     # compute the class loss of features
#     src_class_loss = class_criterion(src_class_out, src_labels)
#     tgt_class_loss = class_criterion(tgt_class_out, tgt_labels)

#     # make prediction based on logits output class_out
#     out_sigmoid = torch.sigmoid(src_class_out).data.detach().cpu().numpy()
#     src_class_pred = np.argmax(out_sigmoid, 1)
#     out_sigmoid = torch.sigmoid(tgt_class_out).data.detach().cpu().numpy()
#     tgt_class_pred = np.argmax(out_sigmoid, 1)


# # #     if i == 0:
# # #       print(out_sigmoid.shape)
# # #       print(tgt_labels, out_sigmoid, tgt_class_pred)
# # #       sys.exit()
# #     print(tgt_labels, out_sigmoid[:,1])
# #     tgt_labels = torch.FloatTensor(out_sigmoid[:,1])
# #     print(tgt_labels, out_sigmoid[:,1])
# #     print('\n')


#     # make prediction based on logits output domain_out
#     out_sigmoid = torch.sigmoid(src_domain_out).data.detach().cpu().numpy()
#     src_domain_pred = np.argmax(out_sigmoid, 1)
#     out_sigmoid = torch.sigmoid(tgt_domain_out).data.detach().cpu().numpy()
#     tgt_domain_pred = np.argmax(out_sigmoid, 1)

#     src_domain_loss = domain_criterion(src_domain_out, src_domain_labels)
#     tgt_domain_loss = domain_criterion(tgt_domain_out, tgt_domain_labels)
#     domain_loss = src_domain_loss + tgt_domain_loss

#     if training_mode == 'dann':
#       theta = 1
#       train_loss = src_class_loss + theta * domain_loss
#     else:
#       train_loss = src_class_loss
	
	
#     # Backward and optimize
#     optimizer.zero_grad()
#     train_loss.backward()
#     optimizer.step()
	
#     total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
#     total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
#     total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
#     total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()

#     domain_TPTN += (src_domain_pred==src_domain_labels.data.detach().cpu().numpy()).sum()
#     domain_TPTN += (tgt_domain_pred==tgt_domain_labels.data.detach().cpu().numpy()).sum()
	
#     src_labels_np = src_labels.data.detach().cpu().numpy()
#     src_TP += ((src_class_pred==1) & (src_labels_np==1)).sum()
#     src_FP += ((src_class_pred==1) & (src_labels_np==0)).sum()
#     src_TN += ((src_class_pred==0) & (src_labels_np==0)).sum()
#     src_FN += ((src_class_pred==0) & (src_labels_np==1)).sum()
	
#     tgt_labels_np = tgt_labels.data.detach().cpu().numpy()
#     tgt_TP += ((tgt_class_pred==1) & (tgt_labels_np==1)).sum()
#     tgt_FP += ((tgt_class_pred==1) & (tgt_labels_np==0)).sum()
#     tgt_TN += ((tgt_class_pred==0) & (tgt_labels_np==0)).sum()
#     tgt_FN += ((tgt_class_pred==0) & (tgt_labels_np==1)).sum()

# #   DATA_train_loader = src_loader.dataset.labels.numpy()
# #   print('SRC sampler: label0, label1', i, src_FP+src_TN, src_TP+src_FN)
# #   print('in SRC train_loader: label0, label1', i, (DATA_train_loader==0).sum(), (DATA_train_loader==1).sum())
# #   sys.exit()

#   src_size = src_loader.dataset.labels.detach().cpu().numpy().shape[0]
#   src_class_loss = total_src_class_loss/src_size
#   src_domain_loss = total_src_domain_loss/src_size
#   src_acc = (src_TP+src_TN)/src_size
#   src_sensitivity = src_TP/(src_TP+src_FN)
#   src_specificity = src_TN/(src_TN+src_FP)

#   src_precision = src_TP/(src_TP+src_FP)
#   if math.isnan(src_precision):
#     if src_FP==0:
#       src_precision=1
#     else:
#       src_precision=0

#   src_F1 = 2 * (src_precision * src_sensitivity) / (src_precision + src_sensitivity)
#   if math.isnan(src_F1):
#     if src_TP + src_FP + src_FN == 0:
#       src_F1 = 1
#     else:
#       src_F1 = 0

#   tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
#   tgt_class_loss = total_tgt_class_loss/tgt_size
#   tgt_domain_loss = total_tgt_domain_loss/tgt_size
#   tgt_acc = (tgt_TP+tgt_TN)/tgt_size
#   tgt_sensitivity = tgt_TP/(tgt_TP+tgt_FN)
#   tgt_specificity = tgt_TN/(tgt_TN+tgt_FP)
#   tgt_precision = tgt_TP/(tgt_TP+tgt_FP)
#   if math.isnan(tgt_precision):
#     if tgt_FP==0:
#       tgt_precision=1
#     else:
#       tgt_precision=0

#   tgt_F1 = 2 * (tgt_precision * tgt_sensitivity) / (tgt_precision + tgt_sensitivity)
#   if math.isnan(tgt_F1):
#     if tgt_TP + tgt_FP + tgt_FN == 0:
#       tgt_F1 = 1
#     else:
#       tgt_F1 = 0

#   domain_acc = domain_TPTN/(src_size+tgt_size)

#   performance_dict = {
#       'src_class_loss': src_class_loss,
#       'src_domain_loss': src_domain_loss,
#       'src_acc': src_acc,
#       'src_sensitivity': src_sensitivity,
#       'src_precision': src_precision,
#       'src_specificity': src_specificity,
#       'src_F1': src_F1,
#       'tgt_class_loss': tgt_class_loss,
#       'tgt_domain_loss': tgt_domain_loss,
#       'tgt_acc': tgt_acc,
#       'tgt_sensitivity': tgt_sensitivity,
#       'tgt_precision': tgt_precision,
#       'tgt_specificity': tgt_specificity,
#       'tgt_F1': tgt_F1,
# 	  'domain_acc': domain_acc,
#   }

#   return performance_dict


def train_epoch_dann(src_loader, tgt_loader, device, dann, class_criterion, domain_criterion, optimizer, training_mode):
  dann.train()

  total_src_class_loss = 0
  total_tgt_class_loss = 0
  total_src_domain_loss = 0
  total_tgt_domain_loss = 0

  domain_TPTN = 0
	
  src_TP = 0
  src_FP = 0
  src_TN = 0
  src_FN = 0
	
  tgt_TP = 0
  tgt_FP = 0
  tgt_TN = 0
  tgt_FN = 0

  for i, (sdata, tdata) in enumerate(zip(src_loader, tgt_loader)):
    src_data, src_labels = sdata
    tgt_data, tgt_labels = tdata

    src_data = src_data.to(device)
    src_labels = src_labels.to(device).long()
    tgt_data = tgt_data.to(device)
    tgt_labels = tgt_labels.to(device).long()

    # prepare domain labels
    src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
    tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

    src_feature, src_class_out, src_domain_out = dann(src_data)
    tgt_feature, tgt_class_out, tgt_domain_out = dann(tgt_data)

    # compute the class loss of features
    src_class_loss = class_criterion(src_class_out, src_labels)
    tgt_class_loss = class_criterion(tgt_class_out, tgt_labels)

    # make prediction based on logits output class_out
    out_sigmoid = torch.sigmoid(src_class_out).data.detach().cpu().numpy()
    src_class_pred = np.argmax(out_sigmoid, 1)
    out_sigmoid = torch.sigmoid(tgt_class_out).data.detach().cpu().numpy()
    tgt_class_pred = np.argmax(out_sigmoid, 1)


# #     if i == 0:
# #       print(out_sigmoid.shape)
# #       print(tgt_labels, out_sigmoid, tgt_class_pred)
# #       sys.exit()
#     print(tgt_labels, out_sigmoid[:,1])
#     tgt_labels = torch.FloatTensor(out_sigmoid[:,1])
#     print(tgt_labels, out_sigmoid[:,1])
#     print('\n')


    # make prediction based on logits output domain_out
    out_sigmoid = torch.sigmoid(src_domain_out).data.detach().cpu().numpy()
    src_domain_pred = np.argmax(out_sigmoid, 1)
    out_sigmoid = torch.sigmoid(tgt_domain_out).data.detach().cpu().numpy()
    tgt_domain_pred = np.argmax(out_sigmoid, 1)

    src_domain_loss = domain_criterion(src_domain_out, src_domain_labels)
    tgt_domain_loss = domain_criterion(tgt_domain_out, tgt_domain_labels)
    domain_loss = src_domain_loss + tgt_domain_loss

    if training_mode == 'dann':
      train_loss = src_class_loss + λ * domain_loss
    else:
      train_loss = src_class_loss
	
	
    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
	
#     total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
#     total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
#     total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
#     total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()

#     domain_TPTN += (src_domain_pred==src_domain_labels.data.detach().cpu().numpy()).sum()
#     domain_TPTN += (tgt_domain_pred==tgt_domain_labels.data.detach().cpu().numpy()).sum()
	
#     src_labels_np = src_labels.data.detach().cpu().numpy()
#     src_TP += ((src_class_pred==1) & (src_labels_np==1)).sum()
#     src_FP += ((src_class_pred==1) & (src_labels_np==0)).sum()
#     src_TN += ((src_class_pred==0) & (src_labels_np==0)).sum()
#     src_FN += ((src_class_pred==0) & (src_labels_np==1)).sum()
	
#     tgt_labels_np = tgt_labels.data.detach().cpu().numpy()
#     tgt_TP += ((tgt_class_pred==1) & (tgt_labels_np==1)).sum()
#     tgt_FP += ((tgt_class_pred==1) & (tgt_labels_np==0)).sum()
#     tgt_TN += ((tgt_class_pred==0) & (tgt_labels_np==0)).sum()
#     tgt_FN += ((tgt_class_pred==0) & (tgt_labels_np==1)).sum()

# #   DATA_train_loader = src_loader.dataset.labels.numpy()
# #   print('SRC sampler: label0, label1', i, src_FP+src_TN, src_TP+src_FN)
# #   print('in SRC train_loader: label0, label1', i, (DATA_train_loader==0).sum(), (DATA_train_loader==1).sum())
# #   sys.exit()

#   src_size = src_loader.dataset.labels.detach().cpu().numpy().shape[0]
#   src_class_loss = total_src_class_loss/src_size
#   src_domain_loss = total_src_domain_loss/src_size
#   src_acc = (src_TP+src_TN)/src_size
#   src_sensitivity = src_TP/(src_TP+src_FN)
#   src_specificity = src_TN/(src_TN+src_FP)

#   src_precision = src_TP/(src_TP+src_FP)
#   if math.isnan(src_precision):
#     if src_FP==0:
#       src_precision=1
#     else:
#       src_precision=0

#   src_F1 = 2 * (src_precision * src_sensitivity) / (src_precision + src_sensitivity)
#   if math.isnan(src_F1):
#     if src_TP + src_FP + src_FN == 0:
#       src_F1 = 1
#     else:
#       src_F1 = 0

#   tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
#   tgt_class_loss = total_tgt_class_loss/tgt_size
#   tgt_domain_loss = total_tgt_domain_loss/tgt_size
#   tgt_acc = (tgt_TP+tgt_TN)/tgt_size
#   tgt_sensitivity = tgt_TP/(tgt_TP+tgt_FN)
#   tgt_specificity = tgt_TN/(tgt_TN+tgt_FP)
#   tgt_precision = tgt_TP/(tgt_TP+tgt_FP)
#   if math.isnan(tgt_precision):
#     if tgt_FP==0:
#       tgt_precision=1
#     else:
#       tgt_precision=0

#   tgt_F1 = 2 * (tgt_precision * tgt_sensitivity) / (tgt_precision + tgt_sensitivity)
#   if math.isnan(tgt_F1):
#     if tgt_TP + tgt_FP + tgt_FN == 0:
#       tgt_F1 = 1
#     else:
#       tgt_F1 = 0

#   domain_acc = domain_TPTN/(src_size+tgt_size)

#   performance_dict = {
#       'src_class_loss': src_class_loss,
#       'src_domain_loss': src_domain_loss,
#       'src_acc': src_acc,
#       'src_sensitivity': src_sensitivity,
#       'src_precision': src_precision,
#       'src_specificity': src_specificity,
#       'src_F1': src_F1,
#       'tgt_class_loss': tgt_class_loss,
#       'tgt_domain_loss': tgt_domain_loss,
#       'tgt_acc': tgt_acc,
#       'tgt_sensitivity': tgt_sensitivity,
#       'tgt_precision': tgt_precision,
#       'tgt_specificity': tgt_specificity,
#       'tgt_F1': tgt_F1,
# 	  'domain_acc': domain_acc,
#   }

#   return performance_dict
  return {}



def val_epoch_dann(src_loader, tgt_loader, device, 
                     dann,
                     class_criterion, domain_criterion, training_mode):

  dann.eval()

  total_val_loss = 0
  total_src_class_loss = 0
  total_tgt_class_loss = 0
  total_src_domain_loss = 0
  total_tgt_domain_loss = 0

  src_class_TPTN = 0
  tgt_class_TPTN = 0
  domain_TPTN = 0
	
  src_TP = 0
  src_FP = 0
  src_TN = 0
  src_FN = 0
	
  tgt_TP = 0
  tgt_FP = 0
  tgt_TN = 0
  tgt_FN = 0

  for i, (sdata, tdata) in enumerate(zip(src_loader, tgt_loader)):

    src_data, src_labels = sdata
    tgt_data, tgt_labels = tdata
	
#     print(src_data[0:5,0,0])
#     print(tgt_data[0:5,0,0])
#     print('\n')

    src_data = src_data.to(device)
    src_labels = src_labels.to(device).long()
    tgt_data = tgt_data.to(device)
    tgt_labels = tgt_labels.to(device).long()

    # prepare domain labels
    src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
    tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

    src_feature, src_class_out, src_domain_out = dann(src_data)
    tgt_feature, tgt_class_out, tgt_domain_out = dann(tgt_data)

    # make prediction based on logits output class_out
    out_sigmoid = torch.sigmoid(src_class_out).data.detach().cpu().numpy()
    src_class_pred = np.argmax(out_sigmoid, 1)
    out_sigmoid = torch.sigmoid(tgt_class_out).data.detach().cpu().numpy()
    tgt_class_pred = np.argmax(out_sigmoid, 1)

    src_class_loss = class_criterion(src_class_out, src_labels)
    tgt_class_loss = class_criterion(tgt_class_out, tgt_labels)


    # make prediction based on logits output domain_out
    out_sigmoid = torch.sigmoid(src_domain_out).data.detach().cpu().numpy()
    src_domain_pred = np.argmax(out_sigmoid, 1)
    out_sigmoid = torch.sigmoid(tgt_domain_out).data.detach().cpu().numpy()
    tgt_domain_pred = np.argmax(out_sigmoid, 1)

    src_domain_loss = domain_criterion(src_domain_out, src_domain_labels)
    tgt_domain_loss = domain_criterion(tgt_domain_out, tgt_domain_labels)
    domain_loss = src_domain_loss + tgt_domain_loss

    if training_mode == 'dann':
      val_loss = src_class_loss + λ * domain_loss
    else:
      val_loss = src_class_loss

    total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
    total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
    total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
    total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()

    src_class_TPTN += (src_class_pred==src_labels.data.detach().cpu().numpy()).sum()
    tgt_class_TPTN += (tgt_class_pred==tgt_labels.data.detach().cpu().numpy()).sum()
    domain_TPTN += (src_domain_pred==src_domain_labels.data.detach().cpu().numpy()).sum()
    domain_TPTN += (tgt_domain_pred==tgt_domain_labels.data.detach().cpu().numpy()).sum()
	
    src_labels_np = src_labels.data.detach().cpu().numpy()
    src_TP += ((src_class_pred==1) & (src_labels_np==1)).sum()
    src_FP += ((src_class_pred==1) & (src_labels_np==0)).sum()
    src_TN += ((src_class_pred==0) & (src_labels_np==0)).sum()
    src_FN += ((src_class_pred==0) & (src_labels_np==1)).sum()
	
    tgt_labels_np = tgt_labels.data.detach().cpu().numpy()
    tgt_TP += ((tgt_class_pred==1) & (tgt_labels_np==1)).sum()
    tgt_FP += ((tgt_class_pred==1) & (tgt_labels_np==0)).sum()
    tgt_TN += ((tgt_class_pred==0) & (tgt_labels_np==0)).sum()
    tgt_FN += ((tgt_class_pred==0) & (tgt_labels_np==1)).sum()


  src_size = src_loader.dataset.labels.detach().cpu().numpy().shape[0]
  src_class_loss = total_src_class_loss/src_size
  src_domain_loss = total_src_domain_loss/src_size
  src_acc = (src_TP+src_TN)/src_size
	
# 	if src_TP+src_FN==0:
	
	
  src_sensitivity = src_TP/(src_TP+src_FN)
  src_specificity = src_TN/(src_TN+src_FP)
  src_precision = src_TP/(src_TP+src_FP)
  if math.isnan(src_precision):
    if src_FP==0:
      src_precision=1
    else:
      src_precision=0

  src_F1 = 2 * (src_precision * src_sensitivity) / (src_precision + src_sensitivity)
  if math.isnan(src_F1):
    if src_TP + src_FP + src_FN == 0:
      src_F1 = 1
    else:
      src_F1 = 0

  tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
  tgt_class_loss = total_tgt_class_loss/tgt_size
  tgt_domain_loss = total_tgt_domain_loss/tgt_size
  tgt_acc = (tgt_TP+tgt_TN)/tgt_size
  tgt_sensitivity = tgt_TP/(tgt_TP+tgt_FN)
  tgt_specificity = tgt_TN/(tgt_TN+tgt_FP)
  tgt_precision = tgt_TP/(tgt_TP+tgt_FP)
  if math.isnan(tgt_precision):
    if tgt_FP==0:
      tgt_precision=1
    else:
      tgt_precision=0

  tgt_F1 = 2 * (tgt_precision * tgt_sensitivity) / (tgt_precision + tgt_sensitivity)
  if math.isnan(tgt_F1):
    if tgt_TP + tgt_FP + tgt_FN == 0:
      tgt_F1 = 1
    else:
      tgt_F1 = 0

  domain_acc = domain_TPTN/(src_size+tgt_size)

  performance_dict = {
      'src_class_loss': src_class_loss,
      'src_domain_loss': src_domain_loss,
      'src_acc': src_acc,
      'src_sensitivity': src_sensitivity,
      'src_precision': src_precision,
      'src_specificity': src_specificity,
      'src_F1': src_F1,
      'tgt_class_loss': tgt_class_loss,
      'tgt_domain_loss': tgt_domain_loss,
      'tgt_acc': tgt_acc,
      'tgt_sensitivity': tgt_sensitivity,
      'tgt_precision': tgt_precision,
      'tgt_specificity': tgt_specificity,
      'tgt_F1': tgt_F1,
	  'domain_acc': domain_acc,
	  'src_class_pred': src_class_pred,
	  'tgt_class_pred': tgt_class_pred
  }

  return performance_dict


def DannModel_fitting(training_params, src_name, tgt_name, i_rep, inputdir, outputdir, training_mode='src'): 
#   show_diagnosis_plt = False

  if not os.path.exists(outputdir):
      os.makedirs(outputdir)

  # TODO: don't need to assign training_params values
  classes_n = training_params['classes_n']
  CV_n = training_params['CV_n']
  num_epochs = training_params['num_epochs']
  channel_n = training_params['channel_n']
  batch_size = training_params['batch_size']
  learning_rate = training_params['learning_rate']
  extractor_type = training_params['extractor_type']
  device = training_params['device']
  show_diagnosis_plt = training_params['show_diagnosis_plt']

  df_performance = pd.DataFrame(0, index=np.arange(CV_n), 
                                columns=['i_CV', 
							 'val_src_acc','val_tgt_acc',
							 'val_src_sensitivity','val_tgt_sensitivity',
							 'val_src_precision','val_tgt_precision',
							 'val_src_F1','val_tgt_F1',
							 'val_domain_acc', 'PAD', 'epoch_optimal'])

	
  src_dataset_name = src_name.split('_')[0]
  src_sensor_loc = src_name.split('_')[1]

  tgt_dataset_name = tgt_name.split('_')[0]
  tgt_sensor_loc = tgt_name.split('_')[1]

  src_inputdir = inputdir + '{}/{}/rep{}/'.format(src_dataset_name, src_sensor_loc, i_rep)
  tgt_inputdir = inputdir + '{}/{}/rep{}/'.format(tgt_dataset_name, tgt_sensor_loc, i_rep)



  for i_CV in range(CV_n):
    print('------------------------------Working on i_CV {}------------------------------'.format(i_CV))
    # 1. prepare dataset
    src_train_loader, src_val_loader, src_train_loader_eval = get_data_loader(src_inputdir, i_CV, batch_size, learning_rate, training_params)
    tgt_train_loader, tgt_val_loader, tgt_train_loader_eval = get_data_loader(tgt_inputdir, i_CV, batch_size, learning_rate, training_params)

    # the model expect the same input dimension for src and tgt data
    src_train_size = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    src_val_size = src_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    tgt_train_size = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    tgt_val_size = tgt_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    src_input_dim = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]
    tgt_input_dim = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]


    # 3. fit the model
    total_step = len(src_train_loader)

    train_performance_dict_list = list( {} for i in range(num_epochs) )
    val_performance_dict_list = list( {} for i in range(num_epochs) )
    train_performance_dict_list_reversed = list( {} for i in range(num_epochs) )
    val_performance_dict_list_reversed = list( {} for i in range(num_epochs) )
    PAD_list = [0] * num_epochs

    if extractor_type == 'CNN':
      model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    elif extractor_type == 'CNNLSTM':
      dropout = training_params['dropout']
      hiddenDim_f = training_params['hiddenDim_f']
      hiddenDim_y = training_params['hiddenDim_y']
      hiddenDim_d = training_params['hiddenDim_d']
      win_size = training_params['win_size']
      win_stride = training_params['win_stride']
      step_n = training_params['step_n']
      model = CnnLstm(device, class_N=classes_n, channel_n=channel_n, dropout=dropout, hiddenDim_f=hiddenDim_f, hiddenDim_y=hiddenDim_y, hiddenDim_d=hiddenDim_d, win_size=win_size, win_stride=win_stride, step_n=step_n).to(device)
        
#     model_reversed = copy.deepcopy(model)
    model_name = model.__class__.__name__
    train_size = src_train_size+tgt_train_size
    # loss and optimizer
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
#     optimizer = optim.AdaMod(
#         m.parameters(),
#         lr= 1e-3,
#         betas=(0.9, 0.999),
#         beta3=0.999,
#         eps=1e-8,
#         weight_decay=0,
#     )


    F1_optimal = 0
    epoch_optimal = -1
	
			
    TGT_train_loader = copy.deepcopy(src_train_loader)
    TGT_train_loader_eval = copy.deepcopy(src_train_loader_eval)
    TGT_val_loader =copy.deepcopy(src_val_loader)

    SRC_train_loader = copy.deepcopy(tgt_train_loader)
    SRC_train_loader_eval = copy.deepcopy(tgt_train_loader_eval)
    SRC_val_loader = copy.deepcopy(tgt_val_loader)

    REV = True
	
    for epoch in range(num_epochs):
#       _ = train_epoch_dann(src_train_loader, tgt_train_loader, device, 
#                                           model, 
#                                           class_criterion, domain_criterion, optimizer, epoch, training_mode)
	
#       train_performance_dict_list[epoch] = val_epoch_dann(src_train_loader_eval, tgt_train_loader_eval, device, 
#                                       model,
#                                       class_criterion, domain_criterion, epoch, training_mode)
	
#       val_performance_dict_list[epoch] = val_epoch_dann(src_val_loader, tgt_val_loader, device, 
#                                       model,
#                                       class_criterion, domain_criterion, epoch, training_mode)

      _ = train_epoch_dann(src_train_loader, tgt_train_loader, device, 
                                          model, 
                                          class_criterion, domain_criterion, optimizer, training_mode)
	
      train_performance_dict_list[epoch] = val_epoch_dann(src_train_loader_eval, tgt_train_loader_eval, device, 
                                      model,
                                      class_criterion, domain_criterion, training_mode)
	
      val_performance_dict_list[epoch] = val_epoch_dann(src_val_loader, tgt_val_loader, device, 
                                      model,
                                      class_criterion, domain_criterion, training_mode)

		
		
		
		
		
#       print('show mod
































































































ch), i_CV, outputdir)
		
#       model_reversed = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
      model_reversed = copy.deepcopy(model)

      SRC_train_loader.dataset.labels = torch.FloatTensor(train_performance_dict_list[epoch]['tgt_class_pred'])
      SRC_train_loader_eval.dataset.labels = torch.FloatTensor(train_performance_dict_list[epoch]['tgt_class_pred'])
      SRC_val_loader.dataset.labels = torch.FloatTensor(val_performance_dict_list[epoch]['tgt_class_pred'])
	
      for epoch_reversed in range(5):
        _ = train_epoch_dann(SRC_train_loader, TGT_train_loader, device, 
											  model_reversed, 
											  class_criterion, domain_criterion, optimizer, training_mode)

      train_performance_dict_list_reversed[epoch] = val_epoch_dann(SRC_train_loader_eval, TGT_train_loader_eval, device, 
										  model_reversed,
										  class_criterion, domain_criterion, training_mode)

      val_performance_dict_list_reversed[epoch]= val_epoch_dann(SRC_val_loader, TGT_val_loader, device, 
										  model_reversed,
										  class_criterion, domain_criterion, training_mode)
		

#       print('show model output')
#       model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)

		
		
		
		
		
      PAD = get_PAD(src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, model, device, c=3000)
      PAD_list[epoch] = PAD
		
		
      # can only use src_F1 since we assume no tgt labels in our scenario
      if training_mode == 'dann':
        if math.isnan(val_performance_dict_list_reversed[epoch]['tgt_F1']):
          continue
        if F1_optimal < val_performance_dict_list_reversed[epoch]['tgt_F1'] or epoch < 5:
          F1_optimal = val_performance_dict_list_reversed[epoch]['tgt_F1']
          epoch_optimal = epoch
          model_optimal = copy.deepcopy(model)
          model_reversed_optimal = copy.deepcopy(model_reversed)
          SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
          SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)
			
      elif training_mode == 'src':
        if math.isnan(val_performance_dict_list_reversed[epoch]['tgt_F1']):
          continue
        if F1_optimal < val_performance_dict_list_reversed[epoch]['tgt_F1'] or epoch < 5:
          F1_optimal = val_performance_dict_list_reversed[epoch]['tgt_F1']
          epoch_optimal = epoch
          model_optimal = copy.deepcopy(model)
          model_reversed_optimal = copy.deepcopy(model_reversed)
          SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
          SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)

      elif training_mode == 'tgt':
        if math.isnan(val_performance_dict_list_reversed[epoch]['tgt_F1']):
          continue
        if F1_optimal < val_performance_dict_list_reversed[epoch]['tgt_F1'] or epoch < 5:
          F1_optimal = val_performance_dict_list_reversed[epoch]['tgt_F1']
          epoch_optimal = epoch
          model_optimal = copy.deepcopy(model)
          model_reversed_optimal = copy.deepcopy(model_reversed)
          SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
          SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)


#       if show_diagnosis_plt:
#         print('show model output')
#         model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)
#         print('show model_reversed output')
#         model_output_diagnosis_trainval(model_reversed, SRC_train_loader, TGT_train_loader, SRC_val_loader, TGT_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)

		
    df_performance.loc[i_CV,['i_CV', 
							 'val_src_acc','val_tgt_acc',
							 'val_src_sensitivity','val_tgt_sensitivity',
							 'val_src_precision','val_tgt_precision',
							 'val_src_F1','val_tgt_F1',
							 'val_domain_acc', 'PAD', 'epoch_optimal']] = [i_CV, 
												   val_performance_dict_list[epoch_optimal]['src_acc'], val_performance_dict_list[epoch_optimal]['tgt_acc'], 
												   val_performance_dict_list[epoch_optimal]['src_sensitivity'], val_performance_dict_list[epoch_optimal]['tgt_sensitivity'], 
												   val_performance_dict_list[epoch_optimal]['src_precision'], val_performance_dict_list[epoch_optimal]['tgt_precision'], 
												   val_performance_dict_list[epoch_optimal]['src_F1'], val_performance_dict_list[epoch_optimal]['tgt_F1'], 
												   val_performance_dict_list[epoch_optimal]['domain_acc'], PAD_list[epoch_optimal], epoch_optimal]
							 
	
    if show_diagnosis_plt:
      dann_learning_diagnosis(num_epochs, train_performance_dict_list, val_performance_dict_list, PAD_list, i_CV, epoch_optimal, outputdir)
      dann_learning_diagnosis(num_epochs, train_performance_dict_list_reversed, val_performance_dict_list_reversed, PAD_list, i_CV, epoch_optimal, outputdir)
    
    print('-----------------Exporting pytorch model-----------------')
    # loaded_model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    if extractor_type == 'CNN':
      loaded_model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    elif extractor_type == 'CNNLSTM':
      dropout = training_params['dropout']
      hiddenDim_f = training_params['hiddenDim_f']
      hiddenDim_y = training_params['hiddenDim_y']
      hiddenDim_d = training_params['hiddenDim_d']
      win_size = training_params['win_size']
      win_stride = training_params['win_stride']
      step_n = training_params['step_n']
      loaded_model = CnnLstm(device, class_N=classes_n, channel_n=channel_n, dropout=dropout, hiddenDim_f=hiddenDim_f, hiddenDim_y=hiddenDim_y, hiddenDim_d=hiddenDim_d, win_size=win_size, win_stride=win_stride, step_n=step_n).to(device)

    loaded_model.eval()
    export_model(model_optimal, loaded_model, outputdir+'model_CV{}'.format(i_CV))

    print('-----------------Evaluating trained model-----------------')
    if show_diagnosis_plt:
      model_output_diagnosis_trainval(loaded_model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch_optimal), i_CV, outputdir)
      model_output_diagnosis_trainval(model_reversed_optimal, SRC_train_loader_optimal, TGT_train_loader, SRC_val_loader_optimal, TGT_val_loader, device, '_epoch{}'.format(epoch_optimal), i_CV, outputdir)

#       model_features_diagnosis_trainval(loaded_model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch_optimal), i_CV, outputdir)

  # 5. export model performance as df
  print('---------------Exporting model performance---------------')
  export_perofmance(df_performance, CV_n, outputdir)

  # 6. export notebook parameters as dict
  # datetime object containing current date and time
  print('--------------Exporting notebook parameters--------------')
  now = datetime.now()
  dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
  samples_n = src_train_size + src_val_size
  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  param_dict = {
      'num_params': num_params,
      'i_rep': i_rep,
      'CV_n': CV_n,
      'samples_n': samples_n,
      'classes_n': classes_n,
      'model_name': model_name,
      'src_dataset_name': src_dataset_name,
      'tgt_dataset_name': tgt_dataset_name,
      'src_sensor_loc': src_sensor_loc,
      'tgt_sensor_loc': tgt_sensor_loc,
      'date': dt_string,
      'num_epochs': num_epochs,
      'channel_n': channel_n,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'input_dim': (batch_size, src_train_loader.dataset.data.size()[1], src_train_loader.dataset.data.size()[2]),
      'output_dim': 2,
      'label_dim': src_train_loader.dataset.labels[0:batch_size].data.detach().cpu().numpy().shape,
	  'epoch_optimal': epoch_optimal
  }
  print(param_dict)

  with open(outputdir+'notebook_param.json', 'w') as fp:
    json.dump(param_dict, fp)


  return df_performance, num_params



def performance_table(src_name, tgt_name, training_params, i_rep, inputdir, outputdir):

  task_name = src_name+'_'+tgt_name

  start_time = time.time()
  print('\n==========================================================================================================================')
  print('======================  train on source, val on target(source={} to target={})  ======================'.format(src_name, tgt_name))
  print('==========================================================================================================================\n')
  source_outputs = DannModel_fitting(training_params, src_name, tgt_name, i_rep, inputdir, outputdir+'source/rep{}/'.format(i_rep), training_mode='src')


  print('\n==========================================================================================================================')
  print('======================  DANN training transferring knowledge(source={} to target={})  ======================'.format(src_name, tgt_name))
  print('==========================================================================================================================\n')
  dann_outputs = DannModel_fitting(training_params, src_name, tgt_name, i_rep, inputdir, outputdir+'dann/rep{}/'.format(i_rep), training_mode='dann')
	
  print('\n==========================================================================================================================')
  print('======================  train on target, val on target(source={} to target={})  ======================'.format(src_name, tgt_name))
  print('==========================================================================================================================\n')
  target_outputs = DannModel_fitting(training_params, tgt_name, src_name, i_rep, inputdir, outputdir+'target/rep{}/'.format(i_rep), training_mode='tgt')
#   target_outputs = BaselineModel_fitting(training_params, tgt_name, src_name, i_rep, inputdir, outputdir+'target/rep{}/'.format(i_rep))



  time_elapsed = time.time() - start_time

  print('time elapsed:', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

  df_performance_src, num_params = source_outputs
  df_performance_tgt, num_params = target_outputs
  df_performance_dann, num_params = dann_outputs

	
  def get_df_performance_table(df_performance_dann, df_performance_src, df_performance_tgt, training_params, metric_name, time_elapsed):
	
    df_performance_table = pd.DataFrame('', index=['channel_n', 'batch_size', 'learning_rate', 
                                                  'source', 'DANN', 'target', 'domain', 'PAD_source', 'PAD_DANN', 'time_elapsed', 'num_params'], columns=[])

    df_performance_table.loc['channel_n',training_params['HP_name']] = training_params['channel_n']
    df_performance_table.loc['batch_size',training_params['HP_name']] = training_params['batch_size']
    df_performance_table.loc['learning_rate',training_params['HP_name']] = training_params['learning_rate']
    df_performance_table.loc['source',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_src.loc['mean']['val_tgt_{}'.format(metric_name)], df_performance_src.loc['std']['val_tgt_{}'.format(metric_name)])
    df_performance_table.loc['DANN',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_dann.loc['mean']['val_tgt_{}'.format(metric_name)], df_performance_dann.loc['std']['val_tgt_{}'.format(metric_name)])
    df_performance_table.loc['target',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_tgt.loc['mean']['val_src_{}'.format(metric_name)], df_performance_tgt.loc['std']['val_src_{}'.format(metric_name)])
    df_performance_table.loc['domain',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_dann.loc['mean']['val_domain_acc'], df_performance_dann.loc['std']['val_domain_acc'])
    df_performance_table.loc['PAD_source',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_src.loc['mean']['PAD'], df_performance_dann.loc['std']['PAD'])
    df_performance_table.loc['PAD_DANN',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_dann.loc['mean']['PAD'], df_performance_dann.loc['std']['PAD'])
    df_performance_table.loc['time_elapsed',training_params['HP_name']] = time_elapsed
    df_performance_table.loc['num_params',training_params['HP_name']] = num_params
	
    return df_performance_table


# TODO combine the get function and dict
  df_performance_table_acc = get_df_performance_table(df_performance_dann, df_performance_src, df_performance_tgt, training_params, 'acc', time_elapsed)
  df_performance_table_sensitivity = get_df_performance_table(df_performance_dann, df_performance_src, df_performance_tgt, training_params, 'sensitivity', time_elapsed)
  df_performance_table_precision = get_df_performance_table(df_performance_dann, df_performance_src, df_performance_tgt, training_params, 'precision', time_elapsed)
  df_performance_table_F1 = get_df_performance_table(df_performance_dann, df_performance_src, df_performance_tgt, training_params, 'F1', time_elapsed)


  df_dict = {
    'df_acc': df_performance_table_acc,
    'df_sensitivity': df_performance_table_sensitivity,
    'df_precision': df_performance_table_precision,
    'df_F1': df_performance_table_F1,
  }    
	
  torch.cuda.empty_cache()

    
  return df_dict
