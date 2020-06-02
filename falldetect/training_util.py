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


def train_epoch_dann(src_loader, tgt_loader, device, dann, class_criterion, domain_criterion, optimizer, λ, training_mode):
  dann.train()

  total_train_loss = 0
  total_src_class_loss = 0
  total_tgt_class_loss = 0
  total_src_domain_loss = 0
  total_tgt_domain_loss = 0

  src_size = 0
  tgt_size = 0

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

    src_size += src_labels.size()[0] 
    tgt_size += tgt_labels.size()[0] 
	
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
    src_class_sigmoid = torch.sigmoid(src_class_out).data.detach().cpu().numpy()
    src_class_pred = np.argmax(src_class_sigmoid, 1)
    tgt_class_sigmoid = torch.sigmoid(tgt_class_out).data.detach().cpu().numpy()
    tgt_class_pred = np.argmax(tgt_class_sigmoid, 1)

    # make prediction based on logits output domain_out
    src_domain_sigmoid = torch.sigmoid(src_domain_out).data.detach().cpu().numpy()
    src_domain_pred = np.argmax(src_domain_sigmoid, 1)
    tgt_domain_sigmoid = torch.sigmoid(tgt_domain_out).data.detach().cpu().numpy()
    tgt_domain_pred = np.argmax(tgt_domain_sigmoid, 1)

    src_domain_loss = domain_criterion(src_domain_out, src_domain_labels)
    tgt_domain_loss = domain_criterion(tgt_domain_out, tgt_domain_labels)
    domain_loss = src_domain_loss + tgt_domain_loss

    if training_mode == 'dann':
      train_loss = src_class_loss + λ * domain_loss
      total_size = src_labels.size()[0] + tgt_labels.size()[0]
    else:
      train_loss = src_class_loss
      total_size = src_labels.size()[0]

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
	
    total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
    total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
    total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
    total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()
    total_train_loss += train_loss.data.detach().cpu().numpy()

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

  src_class_loss = total_src_class_loss/src_size
  src_domain_loss = total_src_domain_loss/src_size
  src_acc = (src_TP+src_TN)/src_size
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

#   tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
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

#   domain_acc = domain_TPTN/(src_size+tgt_size)
#   total_loss = total_train_loss/(src_size+tgt_size)
  domain_acc = domain_TPTN/(src_size+tgt_size)
  total_loss = total_train_loss/total_size

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
	  'total_loss': total_loss,
  }

  return performance_dict

def val_epoch_dann(src_loader, tgt_loader, device, 
                     dann,
                     class_criterion, domain_criterion, λ, training_mode):

  dann.eval()

  total_val_loss = 0
  total_src_class_loss = 0
  total_tgt_class_loss = 0
  total_src_domain_loss = 0
  total_tgt_domain_loss = 0

  src_class_TPTN = 0
  tgt_class_TPTN = 0
  domain_TPTN = 0

  src_size = 0
  tgt_size = 0
	
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
    

    
    
    
    
	
#     print(src_data.size(), tgt_data.size())
#     sys.exit()
    
    src_size += src_labels.size()[0] 
    tgt_size += tgt_labels.size()[0] 

    src_data = src_data.to(device)
    src_labels = src_labels.to(device).long()
    tgt_data = tgt_data.to(device)
    tgt_labels = tgt_labels.to(device).long()

    # prepare domain labels
    src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
    tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

    src_feature, src_class_out, src_domain_out = dann(src_data)
    tgt_feature, tgt_class_out, tgt_domain_out = dann(tgt_data)
    
    

    
        
    
#     aaa = src_loader.dataset.data
#     aaa = aaa.to(device)

#     _, aaa_class_out, _ = dann(aaa)

# #     print('src_loader data:', aaa)
# #     print('src_data:', src_data)
#     print('diff:', (aaa_class_out-src_class_out).sum())
#     sys.exit()

    

    # make prediction based on logits output class_out
    src_class_sigmoid = torch.sigmoid(src_class_out).data.detach().cpu().numpy()
    src_class_pred = np.argmax(src_class_sigmoid, 1)
    tgt_class_sigmoid = torch.sigmoid(tgt_class_out).data.detach().cpu().numpy()
    tgt_class_pred = np.argmax(tgt_class_sigmoid, 1)

    src_class_loss = class_criterion(src_class_out, src_labels)
    tgt_class_loss = class_criterion(tgt_class_out, tgt_labels)


    # make prediction based on logits output domain_out
    src_domain_sigmoid = torch.sigmoid(src_domain_out).data.detach().cpu().numpy()
    src_domain_pred = np.argmax(src_domain_sigmoid, 1)
    tgt_domain_sigmoid = torch.sigmoid(tgt_domain_out).data.detach().cpu().numpy()
    tgt_domain_pred = np.argmax(tgt_domain_sigmoid, 1)

    src_domain_loss = domain_criterion(src_domain_out, src_domain_labels)
    tgt_domain_loss = domain_criterion(tgt_domain_out, tgt_domain_labels)
    domain_loss = src_domain_loss + tgt_domain_loss

    if training_mode == 'dann':
      val_loss = src_class_loss + λ * domain_loss
      total_size = src_labels.size()[0] + tgt_labels.size()[0]
    else:
      val_loss = src_class_loss
      total_size = src_labels.size()[0]

    total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
    total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
    total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
    total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()
    total_val_loss += val_loss.data.detach().cpu().numpy()

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


#   src_size = src_loader.dataset.labels.detach().cpu().numpy().shape[0]
  src_class_loss = total_src_class_loss/src_size
  src_domain_loss = total_src_domain_loss/src_size
  src_acc = (src_TP+src_TN)/src_size
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

#   tgt_size = tgt_loader.dataset.labels.detach().cpu().numpy().shape[0]
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

#   domain_acc = domain_TPTN/(src_size+tgt_size)
#   total_loss = total_val_loss/(src_size+tgt_size)
  domain_acc = domain_TPTN/(src_size+tgt_size)
  total_loss = total_val_loss/total_size

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
	  'total_loss': total_loss,
      'tgt_class_sigmoid': tgt_class_sigmoid[:,1],
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
  λ = training_params['λ']
  extractor_type = training_params['extractor_type']
  device = training_params['device']
  show_diagnosis_plt = training_params['show_diagnosis_plt']

  df_performance = pd.DataFrame(0, index=np.arange(CV_n), 
                                columns=['i_CV', 
							 'val_src_acc','val_tgt_acc',
							 'val_src_sensitivity','val_tgt_sensitivity',
							 'val_src_precision','val_tgt_precision',
							 'val_src_F1','val_tgt_F1',
							 'val_domain_acc', 'PAD', 'epoch_optimal', 'total_loss', 'val_tgt_F1_reversed'])

	
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

    SRC_train_loader, SRC_val_loader, SRC_train_loader_eval = get_data_loader(tgt_inputdir, i_CV, batch_size, learning_rate, training_params)
    TGT_train_loader, TGT_val_loader, TGT_train_loader_eval = get_data_loader(src_inputdir, i_CV, batch_size, learning_rate, training_params)
    

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
    
    num_epoch_reversed = 5
    train_performance_dict_list_reversed = list( {} for i in range(num_epoch_reversed) )
    val_performance_dict_list_reversed = list( {} for i in range(num_epoch_reversed) )
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
        
    skeletal_model = copy.deepcopy(model)
    # model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    model_name = model.__class__.__name__
    train_size = src_train_size+tgt_train_size
    # loss and optimizer
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
	
#     TGT_train_loader = copy.deepcopy(src_train_loader)	
#     TGT_train_loader_eval = copy.deepcopy(src_train_loader_eval)
#     TGT_val_loader =copy.deepcopy(src_val_loader)

#     SRC_train_loader = copy.deepcopy(tgt_train_loader)
#     SRC_train_loader_eval = copy.deepcopy(tgt_train_loader_eval)
#     SRC_val_loader = copy.deepcopy(tgt_val_loader)

#     print(src_train_loader.dataset.labels.sum(), TGT_train_loader.dataset.labels.sum())
#     print(tgt_train_loader.dataset.labels.sum(), SRC_train_loader.dataset.labels.sum())
#     sys.exit()

    REV = True

    epoch_limit = 1

    criterion_name = 'tgt_F1'
    F1_optimal = 0
    epoch_optimal = -1

    for epoch in range(num_epochs):
      _ = train_epoch_dann(src_train_loader, tgt_train_loader, device, 
                                          model, 
                                          class_criterion, domain_criterion, optimizer, λ, training_mode)
	
      train_performance_dict_list[epoch] = val_epoch_dann(src_train_loader_eval, tgt_train_loader_eval, device, 
                                      model,
                                      class_criterion, domain_criterion, λ, training_mode)
	
      val_performance_dict_list[epoch] = val_epoch_dann(src_val_loader, tgt_val_loader, device, 
                                      model,
                                      class_criterion, domain_criterion, λ, training_mode)

		
#       display(val_performance_dict_list[epoch])
#       sys.exit()

#       model_reversed = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
#       model_reversed = copy.deepcopy(model)
      model_reversed = copy.deepcopy(skeletal_model)

      SRC_train_loader.dataset.labels = torch.FloatTensor(train_performance_dict_list[epoch]['tgt_class_pred'])
      SRC_train_loader_eval.dataset.labels = torch.FloatTensor(train_performance_dict_list[epoch]['tgt_class_pred'])
      SRC_val_loader.dataset.labels = torch.FloatTensor(val_performance_dict_list[epoch]['tgt_class_pred'])

#       aaa = train_performance_dict_list[epoch]['tgt_class_pred']
      
      
      
#       aaa = tgt_train_loader.dataset.data
#       aaa = aaa.to(device)  
#       _, aaa_class_out, _ = model(aaa)

#       aaa_class_sigmoid = torch.sigmoid(aaa_class_out).data.detach().cpu().numpy()
#       aaa_class_pred = np.argmax(aaa_class_sigmoid, 1)




#       print('SRC_train_loader labels:', SRC_train_loader.dataset.labels.data.numpy())

#       plt.plot(SRC_train_loader.dataset.labels.data.numpy(),'g', alpha=0.5)

#       SRC_train_loader.dataset.labels = torch.FloatTensor(train_performance_dict_list[epoch]['tgt_class_sigmoid'])
#       SRC_train_loader_eval.dataset.labels = torch.FloatTensor(train_performance_dict_list[epoch]['tgt_class_sigmoid'])
#       SRC_val_loader.dataset.labels = torch.FloatTensor(val_performance_dict_list[epoch]['tgt_class_sigmoid'])
#       SRC_train_loader.dataset.labels = torch.zeros(train_performance_dict_list[epoch]['tgt_class_pred'].shape[0]).float()
#       SRC_train_loader_eval.dataset.labels = torch.zeros(train_performance_dict_list[epoch]['tgt_class_pred'].shape[0]).float()
#       SRC_val_loader.dataset.labels= torch.zeros(val_performance_dict_list[epoch]['tgt_class_pred'].shape[0]).float()
	
  
#       plt.plot(SRC_train_loader.dataset.labels.data.numpy(),'r', alpha=0.3)
#       plt.ylim([0, 1])
#       plt.show()
#       print('SRC_train_loader labels:', SRC_train_loader.dataset.labels.data.numpy())
    
    
#       print(SRC_train_loader.dataset.labels)
#       print(tgt_train_loader.dataset.labels)
#       print(SRC_train_loader.dataset.labels==tgt_train_loader.dataset.labels)
#       sys.exit()

      F1_reversed_stopped = 0
      epoch_reversed_stopped = -1

      for epoch_reversed in range(num_epoch_reversed):
        _ = train_epoch_dann(SRC_train_loader, TGT_train_loader, device, 
                                              model_reversed, 
                                              class_criterion, domain_criterion, optimizer, λ, training_mode)
        train_performance_dict_list_reversed[epoch_reversed] = val_epoch_dann(SRC_train_loader_eval, TGT_train_loader_eval, device, 
										  model_reversed,
										  class_criterion, domain_criterion, λ, training_mode)
        val_performance_dict_list_reversed[epoch_reversed]= val_epoch_dann(SRC_val_loader, TGT_val_loader, device, 
										  model_reversed,
										  class_criterion, domain_criterion, λ, training_mode)
        
        if F1_reversed_stopped < val_performance_dict_list_reversed[epoch_reversed][criterion_name] or epoch_reversed < epoch_limit:
          F1_reversed_stopped = val_performance_dict_list_reversed[epoch_reversed][criterion_name]
          epoch_reversed_stopped = epoch_reversed
          model_reversed_stopped = copy.deepcopy(model_reversed)

        


      
      
#       train_performance_dict_list_reversed[epoch] = val_epoch_dann(SRC_train_loader_eval, TGT_train_loader_eval, device, 
# 										  model_reversed,
# 										  class_criterion, domain_criterion, λ, training_mode)

#       val_performance_dict_list_reversed[epoch]= val_epoch_dann(SRC_val_loader, TGT_val_loader, device, 
# 										  model_reversed,
# 										  class_criterion, domain_criterion, λ, training_mode)
		


		
      PAD = get_PAD(src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, model, device, c=3000)
      PAD_list[epoch] = PAD
		
      if F1_optimal < F1_reversed_stopped or epoch < epoch_limit:
        F1_optimal = F1_reversed_stopped
        epoch_optimal = epoch
        model_optimal = copy.deepcopy(model)
        model_reversed_optimal = copy.deepcopy(model_reversed_stopped)
        SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
        SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)
        
      if show_diagnosis_plt:
        print('show model output')
        model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, '_epoch{}'.format(epoch), i_CV, outputdir)
        print('show model_reversed output')
        model_output_diagnosis_trainval(model_reversed_optimal, SRC_train_loader, TGT_train_loader, SRC_val_loader, TGT_val_loader, device, '_epoch{}_reversed'.format(epoch_reversed_stopped), i_CV, outputdir)

          
#       # can only use src_F1 since we assume no tgt labels in our scenario
#       if training_mode == 'dann':
# #         if math.isnan(val_performance_dict_list_reversed[epoch]['tgt_F1']):
# #           continue
#         if F1_optimal < F1_reversed_optimal or epoch < epoch_limit:
#           F1_optimal = val_performance_dict_list_reversed[epoch_reversed][criterion_name]
#           epoch_optimal = epoch
#           model_optimal = copy.deepcopy(model)
#           model_reversed_optimal = copy.deepcopy(model_reversed)
#           SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
#           SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)
			
#       elif training_mode == 'src':
# #         if math.isnan(val_performance_dict_list_reversed[epoch]['tgt_F1']):
# #           continue
#         if F1_optimal < F1_reversed_optimal or epoch < epoch_limit:
#           F1_optimal = F1_reversed_optimal
#           epoch_optimal = epoch
#           model_optimal = copy.deepcopy(model)
#           model_reversed_optimal = copy.deepcopy(model_reversed)
#           SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
#           SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)

#       elif training_mode == 'tgt':
# #         if math.isnan(val_performance_dict_list_reversed[epoch]['tgt_F1']):
# #           continue
#         if F1_optimal < F1_reversed_optimal or epoch < epoch_limit:
#           F1_optimal = F1_reversed_optimal
#           epoch_optimal = epoch
#           model_optimal = copy.deepcopy(model)
#           model_reversed_optimal = copy.deepcopy(model_reversed)
#           SRC_train_loader_optimal = copy.deepcopy(SRC_train_loader)
#           SRC_val_loader_optimal = copy.deepcopy(SRC_val_loader)



		
    df_performance.loc[i_CV,['i_CV', 
							 'val_src_acc','val_tgt_acc',
							 'val_src_sensitivity','val_tgt_sensitivity',
							 'val_src_precision','val_tgt_precision',
							 'val_src_F1','val_tgt_F1',
							 'val_domain_acc', 'PAD', 'epoch_optimal', 'val_tgt_F1_reversed', 'total_loss']] = [i_CV, 
												   val_performance_dict_list[epoch_optimal]['src_acc'], val_performance_dict_list[epoch_optimal]['tgt_acc'], 
												   val_performance_dict_list[epoch_optimal]['src_sensitivity'], val_performance_dict_list[epoch_optimal]['tgt_sensitivity'], 
												   val_performance_dict_list[epoch_optimal]['src_precision'], val_performance_dict_list[epoch_optimal]['tgt_precision'], 
												   val_performance_dict_list[epoch_optimal]['src_F1'], val_performance_dict_list[epoch_optimal]['tgt_F1'], 
												   val_performance_dict_list[epoch_optimal]['domain_acc'], PAD_list[epoch_optimal], epoch_optimal, val_performance_dict_list_reversed[epoch_optimal][criterion_name], val_performance_dict_list[epoch_optimal]['total_loss']]
							 
	
    if show_diagnosis_plt:
      metric_list = ['total_loss', 'class_loss', 'domain_loss', 'acc', 'sensitivity', 'precision', 'F1', 'PAD']
      dann_learning_diagnosis(num_epochs, train_performance_dict_list, val_performance_dict_list, PAD_list, i_CV, epoch_optimal, metric_list, outputdir)
      metric_list = ['total_loss', 'class_loss', 'domain_loss', 'acc', 'sensitivity', 'precision', 'F1']
      dann_learning_diagnosis(num_epochs, train_performance_dict_list_reversed, val_performance_dict_list_reversed, PAD_list, i_CV, epoch_reversed_stopped, metric_list, outputdir)
    
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
      model_output_diagnosis_trainval(model_reversed_optimal, SRC_train_loader_optimal, TGT_train_loader, SRC_val_loader_optimal, TGT_val_loader, device, '_epoch{}_reversed'.format(epoch_optimal), i_CV, outputdir)

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
