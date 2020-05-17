import numpy as np

import pandas as pd
pd.set_option('display.max_columns', 500)
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
import os
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

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )

from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


def train_epoch(train_loader, device, model, criterion, optimizer, epoch):
  debug = False

  model.train()
  total_train_loss = 0

  TP = 0
  FP = 0
  TN = 0
  FN = 0

  for i, (data, labels) in enumerate(train_loader):

    data = data.to(device)
    labels = labels.to(device).long()

    # Forward pass
    # feature_out, class_out = model(data)
    feature_out, class_out, _ = model(data)
    train_loss = criterion(class_out, labels)

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # total_train_loss += train_loss.data.numpy()
    total_train_loss += train_loss.data.detach().cpu().numpy()
    out_sigmoid = torch.sigmoid(class_out).data.detach().cpu().numpy()
    pred = np.argmax(out_sigmoid, 1)
    labels_np = labels.data.detach().cpu().numpy()

    TP += ((pred==1) & (labels_np==1)).sum()
    FP += ((pred==1) & (labels_np==0)).sum()
    TN += ((pred==0) & (labels_np==0)).sum()
    FN += ((pred==0) & (labels_np==1)).sum()

  train_size = train_loader.dataset.labels.detach().cpu().numpy().shape[0]
  train_loss = total_train_loss/train_size
  acc = (TP+TN)/train_size
  sensitivity = TP/(TP+FN)
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

  if debug:
    print('show train_epoch results')
    print('all samples n', train_size)
    print('TP', TP)
    print('FP', FP)
    print('TN', TN)
    print('FN', FN)
    print('train_size', train_size)
    print('acc', acc)
    print('sensitivity', sensitivity)
    print('precision', precision)
    print('specificity', specificity)
    print('F1', F1)

  performance_dict = {
      'loss': train_loss,
      'acc': acc,
      'sensitivity': sensitivity,
      'precision': precision,
      'specificity': specificity,
      'F1': F1,
  }
  return performance_dict


def val_epoch(val_loader, device, model, criterion, optimizer, epoch, domain_name):
  debug = False

  model.eval()

  total_val_loss = 0
  TP = 0
  FP = 0
  TN = 0
  FN = 0
  for i, (data, labels) in enumerate(val_loader):
    data = data.to(device)
    labels = labels.to(device).long()
    
    #Forward pass
    # feature_out, class_out = model(data)
    feature_out, class_out, _ = model(data)
    val_loss = criterion(class_out, labels)
    
    total_val_loss += val_loss.data.detach().cpu().numpy()
    out_sigmoid = torch.sigmoid(class_out).data.detach().cpu().numpy()
    pred = np.argmax(out_sigmoid, 1)
    labels_np = labels.data.detach().cpu().numpy()

    TP += ((pred==1) & (labels_np==1)).sum()
    FP += ((pred==1) & (labels_np==0)).sum()
    TN += ((pred==0) & (labels_np==0)).sum()
    FN += ((pred==0) & (labels_np==1)).sum()

  val_size = val_loader.dataset.labels.detach().cpu().numpy().shape[0]
  val_loss = total_val_loss/val_size
  acc = (TP+TN)/val_size
  sensitivity = TP/(TP+FN)
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

  if debug:
    print('show val_epoch results')
    print('all samples n', val_size)
    print('TP', TP)
    print('FP', FP)
    print('TN', TN)
    print('FN', FN)
    print('val_size', val_size)
    print('acc', acc)
    print('sensitivity', sensitivity)
    print('precision', precision)
    print('specificity', specificity)
    print('F1', F1)

  performance_dict = {
      '{}_loss'.format(domain_name): val_loss,
      '{}_acc'.format(domain_name): acc,
      '{}_sensitivity'.format(domain_name): sensitivity,
      '{}_precision'.format(domain_name): precision,
      '{}_specificity'.format(domain_name): specificity,
      '{}_F1'.format(domain_name): F1,
  }
  return performance_dict

def train_epoch_dann(src_loader, tgt_loader, device, dann, class_criterion, domain_criterion, optimizer, epoch):
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

    # make prediction based on logits output domain_out
    out_sigmoid = torch.sigmoid(src_domain_out).data.detach().cpu().numpy()
    src_domain_pred = np.argmax(out_sigmoid, 1)
    out_sigmoid = torch.sigmoid(tgt_domain_out).data.detach().cpu().numpy()
    tgt_domain_pred = np.argmax(out_sigmoid, 1)

    src_domain_loss = domain_criterion(src_domain_out, src_domain_labels)
    tgt_domain_loss = domain_criterion(tgt_domain_out, tgt_domain_labels)
    domain_loss = src_domain_loss + tgt_domain_loss

    theta = 1
    train_loss = src_class_loss + theta * domain_loss

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
	
    total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
    total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
    total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
    total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()

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
  src_class_acc = (src_TP+src_TN)/src_size
  src_sensitivity = src_TP/(src_TP+src_FN)
  src_specificity = src_TN/(src_TN+src_FP)
  src_precision = src_TP/(src_TP+src_FP)
  src_F1 = 2 * (src_precision * src_sensitivity) / (src_precision + src_sensitivity)

  tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
  tgt_class_loss = total_tgt_class_loss/tgt_size
  tgt_domain_loss = total_tgt_domain_loss/tgt_size
  tgt_class_acc = (tgt_TP+tgt_TN)/tgt_size
  tgt_sensitivity = tgt_TP/(tgt_TP+tgt_FN)
  tgt_specificity = tgt_TN/(tgt_TN+tgt_FP)
  tgt_precision = tgt_TP/(tgt_TP+tgt_FP)
  tgt_F1 = 2 * (tgt_precision * tgt_sensitivity) / (tgt_precision + tgt_sensitivity)

  domain_acc = domain_TPTN/(src_size+tgt_size)

  performance_dict = {
      'src_class_loss': src_class_loss,
      'src_domain_loss': src_domain_loss,
      'src_class_acc': src_class_acc,
      'src_sensitivity': src_sensitivity,
      'src_precision': src_precision,
      'src_specificity': src_specificity,
      'src_F1': src_F1,
      'tgt_class_loss': tgt_class_loss,
      'tgt_domain_loss': tgt_domain_loss,
      'tgt_class_acc': tgt_class_acc,
      'tgt_sensitivity': tgt_sensitivity,
      'tgt_precision': tgt_precision,
      'tgt_specificity': tgt_specificity,
      'tgt_F1': tgt_F1,
	  'domain_acc': domain_acc,
  }

  return performance_dict

def val_epoch_dann(src_loader, tgt_loader, device, 
                     dann,
                     class_criterion, domain_criterion, epoch):

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
  # for i, sdata in enumerate(src_loader):
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

    theta = 1
    val_loss = src_class_loss + theta * domain_loss

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
  src_class_acc = (src_TP+src_TN)/src_size
  src_sensitivity = src_TP/(src_TP+src_FN)
  src_specificity = src_TN/(src_TN+src_FP)
  src_precision = src_TP/(src_TP+src_FP)
  src_F1 = 2 * (src_precision * src_sensitivity) / (src_precision + src_sensitivity)

  tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
  tgt_class_loss = total_tgt_class_loss/tgt_size
  tgt_domain_loss = total_tgt_domain_loss/tgt_size
  tgt_class_acc = (tgt_TP+tgt_TN)/tgt_size
  tgt_sensitivity = tgt_TP/(tgt_TP+tgt_FN)
  tgt_specificity = tgt_TN/(tgt_TN+tgt_FP)
  tgt_precision = tgt_TP/(tgt_TP+tgt_FP)
  tgt_F1 = 2 * (tgt_precision * tgt_sensitivity) / (tgt_precision + tgt_sensitivity)

  domain_acc = domain_TPTN/(src_size+tgt_size)

  performance_dict = {
      'src_class_loss': src_class_loss,
      'src_domain_loss': src_domain_loss,
      'src_class_acc': src_class_acc,
      'src_sensitivity': src_sensitivity,
      'src_precision': src_precision,
      'src_specificity': src_specificity,
      'src_F1': src_F1,
      'tgt_class_loss': tgt_class_loss,
      'tgt_domain_loss': tgt_domain_loss,
      'tgt_class_acc': tgt_class_acc,
      'tgt_sensitivity': tgt_sensitivity,
      'tgt_precision': tgt_precision,
      'tgt_specificity': tgt_specificity,
      'tgt_F1': tgt_F1,
	  'domain_acc': domain_acc
  }
#   display(performance_dict)
  return performance_dict

#   return val_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc

def BaselineModel_fitting(training_params, src_name, tgt_name, inputdir, outputdir): 
  show_train_log = False
#   show_diagnosis_plt = False

  if not os.path.exists(outputdir):
    os.makedirs(outputdir)
      
  # TODO: don't need to extract training_params
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
                                         'val_src_F1','val_tgt_F1','PAD'])

  src_dataset_name = src_name.split('_')[0]
  src_sensor_loc = src_name.split('_')[1]

  tgt_dataset_name = tgt_name.split('_')[0]
  tgt_sensor_loc = tgt_name.split('_')[1]

  src_inputdir = inputdir + '{}/{}/'.format(src_dataset_name, src_sensor_loc)
  tgt_inputdir = inputdir + '{}/{}/'.format(tgt_dataset_name, tgt_sensor_loc)

  get_src_loader = get_data_loader
  get_tgt_loader = get_data_loader

  for i_CV in range(CV_n):
    print('------------------------------Working on i_CV {}------------------------------'.format(i_CV))
    # 1. prepare dataset
    src_train_loader, src_val_loader = get_src_loader(src_inputdir, i_CV, batch_size, learning_rate)
    tgt_train_loader, tgt_val_loader = get_tgt_loader(tgt_inputdir, i_CV, batch_size, learning_rate)

    # the model expect the same input dimension for src and tgt data
    src_train_size = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    src_val_size = src_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    tgt_train_size = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    tgt_val_size = tgt_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    src_input_dim = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]
    tgt_input_dim = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]

    # 2. prepare model

    total_step = len(src_train_loader)
	
    train_performance_dict_list = list( {} for i in range(num_epochs) )
    val_src_performance_dict_list = list( {} for i in range(num_epochs) )
    val_tgt_performance_dict_list = list( {} for i in range(num_epochs) )

    PAD_list = [0] * num_epochs
	
    # model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
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

    model_name = model.__class__.__name__
    # loss and optimizer
    class_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

#     if show_diagnosis_plt:
#       model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(0), i_CV, outputdir)
#       model_features_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(0), i_CV, outputdir)

    # 3. fit the model
    for epoch in range(num_epochs):
      train_performance_dict = train_epoch(src_train_loader, device, model, class_criterion, optimizer, epoch)
      train_performance_dict_list[epoch] = train_performance_dict

      val_src_performance_dict = val_epoch(src_val_loader, device, model, class_criterion, optimizer, epoch, 'src')
      val_src_performance_dict_list[epoch] = val_src_performance_dict

      val_tgt_performance_dict = val_epoch(tgt_val_loader, device, model, class_criterion, optimizer, epoch, 'tgt')
      val_tgt_performance_dict_list[epoch] = val_tgt_performance_dict
		
      PAD = get_PAD(src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, model, device, c=3000)
      PAD_list[epoch] = PAD

#     fig = plt.figure(figsize=(5, 3), dpi=80)
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_title('PAD')
#     ax1.set_xlabel('epoch')
#     ax1.plot(np.arange(num_epochs), PAD_list, label='PAD')
#     ax1.legend(loc="upper right")


	
    df_performance.loc[i_CV,['i_CV',
							 'val_src_acc','val_tgt_acc',
							 'val_src_sensitivity','val_tgt_sensitivity',
							 'val_src_precision','val_tgt_precision',
							 'val_src_F1','val_tgt_F1', 'PAD']] = [i_CV, 
															val_src_performance_dict_list[epoch]['src_acc'], val_tgt_performance_dict_list[epoch]['tgt_acc'],
															val_src_performance_dict_list[epoch]['src_sensitivity'], val_tgt_performance_dict_list[epoch]['tgt_sensitivity'], 
															val_src_performance_dict_list[epoch]['src_precision'], val_tgt_performance_dict_list[epoch]['tgt_precision'], 
															val_src_performance_dict_list[epoch]['src_F1'], val_tgt_performance_dict_list[epoch]['tgt_F1'], PAD_list[epoch]]
	
    
    if show_diagnosis_plt:
      baseline_learning_diagnosis(num_epochs, train_performance_dict_list, val_src_performance_dict_list, val_tgt_performance_dict_list, PAD_list, i_CV, outputdir)

    print('-----------------Exporting pytorch model-----------------')
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

    export_model(model, loaded_model, outputdir+'model_CV{}'.format(i_CV))

    print('-----------------Evaluating trained model-----------------')
    if show_diagnosis_plt:
      model_output_diagnosis_trainval(loaded_model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)
      model_features_diagnosis_trainval(loaded_model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)

  # 5. export model performance as df
# 	TODO: probably remove this
  print('---------------Exporting model performance---------------')
  export_perofmance(df_performance, CV_n, outputdir)

#   print('src val loss: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_loss'], df_performance.loc['std']['val_loss']))
  print('src val acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_src_acc'], df_performance.loc['std']['val_src_acc']))
  
#   print('tgt val loss: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['tgt_val_loss'], df_performance.loc['std']['tgt_val_loss']))
  print('tgt val acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_tgt_acc'], df_performance.loc['std']['val_tgt_acc']))

  # print('=========================================================')

  # 6. export notebook parameters as dict
  # datetime object containing current date and time
# 	TODO: probably remove this
  print('--------------Exporting notebook parameters--------------')
  now = datetime.now()
  dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
  samples_n = src_train_size + src_val_size

  # TODO: don't need to make param_dict

  param_dict = {
      'CV_n': CV_n,
      'samples_n': samples_n,
      'classes_n': classes_n,
      'model_name': model_name,
      'dataset_name': src_dataset_name,
      'num_epochs': num_epochs,
      'channel_n': channel_n,
      'batch_size': batch_size,
      'learning_rate': learning_rate,
      'sensor_loc': src_sensor_loc,
      'date': dt_string,
      'input_dim': (batch_size, src_train_loader.dataset.data.size()[1], src_train_loader.dataset.data.size()[2]),
      'output_dim': src_train_loader.dataset.labels[0:batch_size].data.detach().cpu().numpy().shape,
  }

  print(param_dict)

  with open(outputdir+'notebook_param.json', 'w') as fp:
    json.dump(param_dict, fp)

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


  return df_performance, num_params

def DannModel_fitting(training_params, src_name, tgt_name, inputdir, outputdir): 
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
							 'val_src_class_acc','val_tgt_class_acc',
							 'val_src_class_sensitivity','val_tgt_class_sensitivity',
							 'val_src_class_precision','val_tgt_class_precision',
							 'val_src_class_F1','val_tgt_class_F1',
							 'val_domain_acc','PAD'])
	
  src_dataset_name = src_name.split('_')[0]
  src_sensor_loc = src_name.split('_')[1]

  tgt_dataset_name = tgt_name.split('_')[0]
  tgt_sensor_loc = tgt_name.split('_')[1]

  src_inputdir = inputdir + '{}/{}/'.format(src_dataset_name, src_sensor_loc)
  tgt_inputdir = inputdir + '{}/{}/'.format(tgt_dataset_name, tgt_sensor_loc)


  get_src_loader = get_data_loader
  get_tgt_loader = get_data_loader
	

  for i_CV in range(CV_n):
    print('------------------------------Working on i_CV {}------------------------------'.format(i_CV))
    # 1. prepare dataset
    src_train_loader, src_val_loader = get_src_loader(src_inputdir, i_CV, batch_size, learning_rate)
    tgt_train_loader, tgt_val_loader = get_tgt_loader(tgt_inputdir, i_CV, batch_size, learning_rate)

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
        
    # model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    model_name = model.__class__.__name__
    train_size = src_train_size+tgt_train_size
    # loss and optimizer
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)

#     if show_diagnosis_plt:
#       model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(0), i_CV, outputdir)
#       model_features_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(0), i_CV, outputdir)

    for epoch in range(num_epochs):
      train_performance_dict_list[epoch] = train_epoch_dann(src_train_loader, tgt_train_loader, device, 
                                          model, 
                                          class_criterion, domain_criterion, optimizer, epoch)
	
      val_performance_dict_list[epoch] = val_epoch_dann(src_val_loader, tgt_val_loader, device, 
                                      model,
                                      class_criterion, domain_criterion, epoch)
      PAD = get_PAD(src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, model, device, c=3000)
      PAD_list[epoch] = PAD

#     fig = plt.figure(figsize=(5, 3), dpi=80)
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.set_title('PAD')
#     ax1.set_xlabel('epoch')
#     ax1.plot(np.arange(num_epochs), PAD_list, label='PAD')
#     ax1.legend(loc="upper right")

    df_performance.loc[i_CV,['i_CV', 
							 'val_src_class_acc','val_tgt_class_acc',
							 'val_src_class_sensitivity','val_tgt_class_sensitivity',
							 'val_src_class_precision','val_tgt_class_precision',
							 'val_src_class_F1','val_tgt_class_F1',
							 'val_domain_acc', 'PAD']] = [i_CV, 
												   val_performance_dict_list[epoch]['src_class_acc'], val_performance_dict_list[epoch]['tgt_class_acc'], 
												   val_performance_dict_list[epoch]['src_sensitivity'], val_performance_dict_list[epoch]['tgt_sensitivity'], 
												   val_performance_dict_list[epoch]['src_precision'], val_performance_dict_list[epoch]['tgt_precision'], 
												   val_performance_dict_list[epoch]['src_F1'], val_performance_dict_list[epoch]['tgt_F1'], 
												   val_performance_dict_list[epoch]['domain_acc'], PAD_list[epoch]]
							 
	
    if show_diagnosis_plt:
      dann_learning_diagnosis(num_epochs, train_performance_dict_list, val_performance_dict_list, PAD_list, i_CV, outputdir)
    
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
    export_model(model, loaded_model, outputdir+'model_CV{}'.format(i_CV))

    print('-----------------Evaluating trained model-----------------')
    if show_diagnosis_plt:
      model_output_diagnosis_trainval(loaded_model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)
      model_features_diagnosis_trainval(loaded_model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)

  # 5. export model performance as df
  print('---------------Exporting model performance---------------')
  export_perofmance(df_performance, CV_n, outputdir)

  # 6. export notebook parameters as dict
  # datetime object containing current date and time
  print('--------------Exporting notebook parameters--------------')
  now = datetime.now()
  dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
  samples_n = src_train_size + src_val_size

  param_dict = {
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
  }
  print(param_dict)

  with open(outputdir+'notebook_param.json', 'w') as fp:
    json.dump(param_dict, fp)

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

  return df_performance, num_params



def performance_table(src_name, tgt_name, training_params, inputdir, outputdir):

  task_name = src_name+'_'+tgt_name

  start_time = time.time()
  print('\n==========================================================================================================================')
  print('======================  train on source, val on target(source={} to target={})  ======================'.format(src_name, tgt_name))
  print('==========================================================================================================================\n')
  source_outputs = BaselineModel_fitting(training_params, src_name, tgt_name, inputdir, outputdir+'source/')
  print('\n==========================================================================================================================')
  print('======================  train on target, val on target(source={} to target={})  ======================'.format(src_name, tgt_name))
  print('==========================================================================================================================\n')
  target_outputs = BaselineModel_fitting(training_params, tgt_name, src_name, inputdir, outputdir+'target/')

  print('\n==========================================================================================================================')
  print('======================  DANN training transferring knowledge(source={} to target={})  ======================'.format(src_name, tgt_name))
  print('==========================================================================================================================\n')
  

  dann_outputs = DannModel_fitting(training_params, src_name, tgt_name, inputdir, outputdir+'dann/')

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
    df_performance_table.loc['DANN',training_params['HP_name']] = '{:.3f}±{:.3f}'.format(df_performance_dann.loc['mean']['val_tgt_class_{}'.format(metric_name)], df_performance_dann.loc['std']['val_tgt_class_{}'.format(metric_name)])
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
