import numpy as np

import pandas as pd
pd.set_option('display.max_columns', 500)
from tqdm import tqdm_notebook as tqdm
from IPython.display import display
import os
import sys
sys.path.append('/content/drive/My Drive/中研院/repo/')

from utilities import *
from models import *
from dataset_util import *

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


def train_epoch(train_loader, train_size, device, model, criterion, optimizer, epoch):
  total_train_loss = 0
  train_TPTF = 0
  for i, (data, labels) in enumerate(train_loader):

    data = data.to(device)
    labels = labels.to(device).long()

    # Forward pass
    feature_out, class_out = model(data)
    train_loss = criterion(class_out, labels)

    # Backward and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # total_train_loss += train_loss.data.numpy()
    total_train_loss += train_loss.data.detach().cpu().numpy()
    out_sigmoid = torch.sigmoid(class_out).data.detach().cpu().numpy()
    train_pred = np.argmax(out_sigmoid, 1)
    train_TPTF += (train_pred==labels.data.detach().cpu().numpy()).sum()

  train_loss = total_train_loss/train_size
  train_acc = train_TPTF/train_size

  return train_loss, train_acc

def val_epoch(val_loader, val_size, device, model, criterion, optimizer, epoch):
  total_val_loss = 0
  val_TPTF = 0
  for i, (data, labels) in enumerate(val_loader):
    data = data.to(device)
    labels = labels.to(device).long()
    
    #Forward pass
    feature_out, class_out = model(data)
    val_loss = criterion(class_out, labels)
    
    total_val_loss += val_loss.data.detach().cpu().numpy()
    out_sigmoid = torch.sigmoid(class_out).data.detach().cpu().numpy()
    val_pred = np.argmax(out_sigmoid, 1)
    val_TPTF += (val_pred==labels.data.detach().cpu().numpy()).sum()

  val_loss = total_val_loss/val_size
  val_acc = val_TPTF/val_size

  return val_loss, val_acc

def train_epoch_dann(src_loader, tgt_loader, src_train_size, tgt_train_size, device, 
                        dann,
                        class_criterion, domain_criterion, optimizer, epoch):
  total_train_loss = 0
  total_src_class_loss = 0
  total_tgt_class_loss = 0
  total_src_domain_loss = 0
  total_tgt_domain_loss = 0

  src_class_TPTN = 0
  tgt_class_TPTN = 0
  domain_TPTN = 0
   # note that this is different from src_train_size as src_loader and 
   # tgt_loader have different sample size
  src_train_count = 0
  tgt_train_count = 0
  train_size = src_train_size + tgt_train_size

  # print('show loader size:', len(src_loader), len(tgt_loader))
  for i, (sdata, tdata) in enumerate(zip(src_loader, tgt_loader)):
  # for i, sdata in enumerate(src_loader):
    src_data, src_labels = sdata
    tgt_data, tgt_labels = tdata
    # print('show data and label size:', src_data.size()[0])
    src_train_count += src_data.size()[0]
    tgt_train_count += tgt_data.size()[0]

    src_data = src_data.to(device)
    src_labels = src_labels.to(device).long()
    tgt_data = tgt_data.to(device)
    tgt_labels = tgt_labels.to(device).long()

    # prepare domain labels
    src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
    tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()
    # print('show domain labels:', src_domain_labels, tgt_domain_labels)


    src_feature, src_class_out, src_domain_out = dann(src_data)
    tgt_feature, tgt_class_out, tgt_domain_out = dann(tgt_data)


    ## CLASS CLASSIFICATION
    # compute the output of source domain and target domain
    # src_feature = feature_extractor(src_data)
    # tgt_feature = feature_extractor(tgt_data)

    # compute the class loss of features
    src_class_loss = class_criterion(src_class_out, src_labels)
    tgt_class_loss = class_criterion(tgt_class_out, tgt_labels)

    # make prediction based on logits output class_out
    out_sigmoid = torch.sigmoid(src_class_out).data.detach().cpu().numpy()
    src_class_pred = np.argmax(out_sigmoid, 1)
    out_sigmoid = torch.sigmoid(tgt_class_out).data.detach().cpu().numpy()
    tgt_class_pred = np.argmax(out_sigmoid, 1)

    # ## DOMAIN CLASSIFICATION
    # # compute the domain loss of src_feature and target_feature
    # src_domain_out = domain_classifier(src_feature, 1)
    # tgt_domain_out = domain_classifier(tgt_feature, 1)

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
    # train_loss = src_class_loss

    # Backward and optimize
    optimizer.zero_grad()
    # train_loss.backward()
    train_loss.backward()
    optimizer.step()

    # total_train_loss += train_loss.data.numpy()
    total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
    total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
    total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
    total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()

    src_class_TPTN += (src_class_pred==src_labels.data.detach().cpu().numpy()).sum()
    tgt_class_TPTN += (tgt_class_pred==tgt_labels.data.detach().cpu().numpy()).sum()
    domain_TPTN += (src_domain_pred==src_domain_labels.data.detach().cpu().numpy()).sum()
    domain_TPTN += (tgt_domain_pred==tgt_domain_labels.data.detach().cpu().numpy()).sum()
    # print(class_pred)
    # print(train_loss)

  # print('total TPTF:', train_TPTF)
  # print('last i:', i)
  # sys.exit()
  train_count = src_train_count + tgt_train_count
  # train_loss_avg = total_train_loss/train_count
  src_class_loss_avg = total_src_class_loss/src_train_count
  tgt_class_loss_avg = total_tgt_class_loss/tgt_train_count

  src_domain_loss_avg = total_src_domain_loss/src_train_count
  tgt_domain_loss_avg = total_tgt_domain_loss/tgt_train_count
  train_loss_avg = src_class_loss_avg + theta * (src_domain_loss_avg + tgt_domain_loss_avg)

  src_class_acc = src_class_TPTN/src_train_count
  tgt_class_acc = tgt_class_TPTN/tgt_train_count
  domain_acc = domain_TPTN/train_count

  return train_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc

def val_epoch_dann(src_loader, tgt_loader, src_val_size, tgt_val_size, device, 
                     dann,
                     class_criterion, domain_criterion, epoch):
  total_val_loss = 0
  total_src_class_loss = 0
  total_tgt_class_loss = 0
  total_src_domain_loss = 0
  total_tgt_domain_loss = 0

  src_class_TPTN = 0
  tgt_class_TPTN = 0
  domain_TPTN = 0
   # note that this is different from src_val_size as src_loader and 
   # tgt_loader have different sample size
  src_val_count = 0
  tgt_val_count = 0
  val_size = src_val_size + tgt_val_size

  # print('show loader size:', len(src_loader), len(tgt_loader))
  for i, (sdata, tdata) in enumerate(zip(src_loader, tgt_loader)):
  # for i, sdata in enumerate(src_loader):
    src_data, src_labels = sdata
    tgt_data, tgt_labels = tdata
    # print('show data and label size:', src_data.size()[0])
    src_val_count += src_data.size()[0]
    tgt_val_count += tgt_data.size()[0]

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

    # total_class_loss += src_class_loss.data.numpy()
    total_src_class_loss += src_class_loss.data.detach().cpu().numpy()
    total_tgt_class_loss += tgt_class_loss.data.detach().cpu().numpy()
    total_src_domain_loss += src_domain_loss.data.detach().cpu().numpy()
    total_tgt_domain_loss += tgt_domain_loss.data.detach().cpu().numpy()

    src_class_TPTN += (src_class_pred==src_labels.data.detach().cpu().numpy()).sum()
    tgt_class_TPTN += (tgt_class_pred==tgt_labels.data.detach().cpu().numpy()).sum()
    domain_TPTN += (src_domain_pred==src_domain_labels.data.detach().cpu().numpy()).sum()
    domain_TPTN += (tgt_domain_pred==tgt_domain_labels.data.detach().cpu().numpy()).sum()


  val_count = src_val_count + tgt_val_count
  src_class_loss_avg = total_src_class_loss/src_val_count
  tgt_class_loss_avg = total_tgt_class_loss/tgt_val_count
  src_domain_loss_avg = total_src_domain_loss/src_val_count
  tgt_domain_loss_avg = total_tgt_domain_loss/tgt_val_count
  val_loss_avg = src_class_loss_avg + theta * (src_domain_loss_avg + tgt_domain_loss_avg)

  src_class_acc = src_class_TPTN/src_val_count
  tgt_class_acc = tgt_class_TPTN/tgt_val_count
  domain_acc = domain_TPTN/val_count
  # print('domain_acc:', domain_TPTN, val_count, domain_acc)
  # sys.exit()

  return val_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc


def BaselineModel_fitting(training_params, src_name, tgt_name, inputdir, outputdir): 
  show_train_log = False

  if not os.path.exists(outputdir):
      os.makedirs(outputdir)
      
  classes_n = training_params['classes_n']
  CV_n = training_params['CV_n']
  num_epochs = training_params['num_epochs']
  channel_n = training_params['channel_n']
  batch_size = training_params['batch_size']
  learning_rate = training_params['learning_rate']
  # dropout_p = training_params['dropout_p']

  df_performance = pd.DataFrame(columns=['i_CV',
                                          'train_loss','train_acc','val_loss','val_acc', 'tgt_val_loss', 'tgt_val_acc'])

  src_dataset_name = src_name.split('_')[0]
  src_sensor_loc = src_name.split('_')[1]

  tgt_dataset_name = tgt_name.split('_')[0]
  tgt_sensor_loc = tgt_name.split('_')[1]

  src_inputdir = inputdir + '{}/{}/'.format(src_dataset_name, src_sensor_loc)
  tgt_inputdir = inputdir + '{}/{}/'.format(tgt_dataset_name, tgt_sensor_loc)


  for i_CV in range(CV_n):
    # 1. prepare dataset
    src_train_loader, src_val_loader = get_UMAFall_loader(src_inputdir, i_CV, batch_size, learning_rate)
    tgt_train_loader, tgt_val_loader = get_UPFall_loader(tgt_inputdir, i_CV, batch_size, learning_rate)

    # the model expect the same input dimension for src and tgt data
    src_train_size = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    src_val_size = src_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    tgt_train_size = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    tgt_val_size = tgt_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    src_input_dim = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]
    tgt_input_dim = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]

    # 2. prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()

    # 3. fit the model
    total_step = len(src_train_loader)

    train_loss_avg_epochs = np.zeros(num_epochs)
    train_class_acc_epochs = np.zeros(num_epochs)
    val_src_loss_avg_epochs = np.zeros(num_epochs)
    val_src_class_acc_epochs = np.zeros(num_epochs)
    val_tgt_loss_avg_epochs = np.zeros(num_epochs)
    val_tgt_class_acc_epochs = np.zeros(num_epochs)

    for epoch in range(num_epochs):

      model = BaselineModel(device, class_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
      model_name = model.__class__.__name__
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
      # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

      train_loss, train_acc = train_epoch(src_train_loader, src_train_size, device, model, class_criterion, optimizer, epoch)
      train_loss_avg_epochs[epoch] = train_loss
      train_class_acc_epochs[epoch] = train_acc

      val_loss, val_acc = val_epoch(src_val_loader, src_val_size, device, model, class_criterion, optimizer, epoch)
      val_src_loss_avg_epochs[epoch] = val_loss
      val_src_class_acc_epochs[epoch] = val_acc

      tgt_val_loss, tgt_val_acc = val_epoch(tgt_val_loader, tgt_val_size, device, model, class_criterion, optimizer, epoch)
      val_tgt_loss_avg_epochs[epoch] = tgt_val_loss
      val_tgt_class_acc_epochs[epoch] = tgt_val_acc

      if show_train_log:
        print('Epoch {}'.format(epoch))
        print('Train Loss: {:.6f}, Train ACC: {:.6f}, Val loss = {:.6f}, Val ACC: {:.6f}'.
              format(train_loss, train_acc, val_loss, val_acc))
        print('Target Val loss = {:.6f}, Val ACC: {:.6f}'.format(tgt_val_loss, tgt_val_acc))

      # 4. store the performance of the model at the last epoch
      df_performance.loc[i_CV] = [i_CV, train_loss, train_acc, val_loss, val_acc, tgt_val_loss, tgt_val_acc]
    
    fig = plt.figure(figsize=(10, 3), dpi=80)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('loss_avg_epochs')
    ax1.set_xlabel('epoch')
    ax1.plot(np.arange(num_epochs), train_loss_avg_epochs, color='blue', label='train')
    ax1.plot(np.arange(num_epochs), val_src_loss_avg_epochs, color='red', label='val_src')
    ax1.plot(np.arange(num_epochs), val_tgt_loss_avg_epochs, color='green', label='val_tgt')
    ax1.legend(loc="upper right")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('class_acc_epochs')
    ax2.set_xlabel('epoch')
    ax2.plot(np.arange(num_epochs), train_class_acc_epochs, color='blue', label='train')
    ax2.plot(np.arange(num_epochs), val_src_class_acc_epochs, color='red', label='val_src')
    ax2.plot(np.arange(num_epochs), val_tgt_class_acc_epochs, color='green', label='val_tgt')
    ax2.legend(loc="upper right")
    plt.show()


    print('=================Exporting pytorch model=================')
    loaded_model = BaselineModel(device, class_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    export_model(model, loaded_model, outputdir+'model_CV{}'.format(i_CV))
    print('=========================================================')

  # 5. export model performance as df
  print('===============Exporting model performance===============')
  export_perofmance(df_performance, CV_n, outputdir)

  print('src val loss: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_loss'], df_performance.loc['std']['val_loss']))
  print('src val acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_acc'], df_performance.loc['std']['val_acc']))
  
  print('tgt val loss: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['tgt_val_loss'], df_performance.loc['std']['tgt_val_loss']))
  print('tgt val acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['tgt_val_acc'], df_performance.loc['std']['tgt_val_acc']))

  print('=========================================================')

  # 6. export notebook parameters as dict
  # datetime object containing current date and time
  print('==============Exporting notebook parameters==============')
  now = datetime.now()
  dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
  samples_n = src_train_size + src_val_size

  param_dict = {
      'CV_n': CV_n,
      'samples_n': samples_n,
      'classes_n': classes_n,
      'model_name': model_name,
      'dataset_name': src_dataset_name,
      'sensor_loc': src_sensor_loc,
      'date': dt_string,
      'batch_size': batch_size,
      'input_dim': (batch_size, src_train_loader.dataset.data.size()[1], src_train_loader.dataset.data.size()[2]),
      'output_dim': src_train_loader.dataset.labels[0:batch_size].data.detach().cpu().numpy().shape,
      'label_dim': CV_n,
  }
  print(param_dict)

  with open(outputdir+'notebook_param.json', 'w') as fp:
    json.dump(param_dict, fp)
  print('=========================================================')

  return (df_performance.loc['mean']['val_acc'], df_performance.loc['std']['val_acc']), (df_performance.loc['mean']['tgt_val_acc'], df_performance.loc['std']['tgt_val_acc'])


def DannModel_fitting(training_params, src_name, tgt_name, inputdir, outputdir): 

  if not os.path.exists(outputdir):
      os.makedirs(outputdir)

  classes_n = training_params['classes_n']
  CV_n = training_params['CV_n']
  num_epochs = training_params['num_epochs']
  channel_n = training_params['channel_n']
  batch_size = training_params['batch_size']
  learning_rate = training_params['learning_rate']

  df_performance = pd.DataFrame(0, index=np.arange(CV_n), 
                                columns=['i_CV',
                                        'train_src_class_loss','train_tgt_class_loss','train_src_domain_loss','train_tgt_domain_loss', 
                                        'train_src_class_acc','train_tgt_class_acc','train_domain_acc',
                                        'val_src_class_loss','val_tgt_class_loss','val_src_domain_loss','val_tgt_domain_loss',
                                        'val_src_class_acc','val_tgt_class_acc','val_domain_acc'])

  src_dataset_name = src_name.split('_')[0]
  src_sensor_loc = src_name.split('_')[1]

  tgt_dataset_name = tgt_name.split('_')[0]
  tgt_sensor_loc = tgt_name.split('_')[1]

  src_inputdir = inputdir + '{}/{}/'.format(src_dataset_name, src_sensor_loc)
  tgt_inputdir = inputdir + '{}/{}/'.format(tgt_dataset_name, tgt_sensor_loc)


  for i_CV in range(CV_n):
    # 1. prepare dataset
    src_train_loader, src_val_loader = get_UMAFall_loader(src_inputdir, i_CV, batch_size, learning_rate)
    tgt_train_loader, tgt_val_loader = get_UPFall_loader(tgt_inputdir, i_CV, batch_size, learning_rate)

    # the model expect the same input dimension for src and tgt data
    src_train_size = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    src_val_size = src_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    tgt_train_size = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[0]
    tgt_val_size = tgt_val_loader.dataset.data.data.detach().cpu().numpy().shape[0]

    src_input_dim = src_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]
    tgt_input_dim = tgt_train_loader.dataset.data.data.detach().cpu().numpy().shape[2]

    # 2. prepare model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    # 3. fit the model
    total_step = len(src_train_loader)

    train_loss_avg_epochs = np.zeros(num_epochs)
    train_src_class_acc_epochs = np.zeros(num_epochs)
    train_tgt_class_acc_epochs = np.zeros(num_epochs)
    train_domain_acc = np.zeros(num_epochs)
    val_loss_avg_epochs = np.zeros(num_epochs)
    val_src_class_acc_epochs = np.zeros(num_epochs)
    val_tgt_class_acc_epochs = np.zeros(num_epochs)
    val_domain_acc = np.zeros(num_epochs)

    for epoch in range(num_epochs):
      # if training_mode == 'dann_v2':
      df_performance.loc[i_CV,'i_CV'] = i_CV
      model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
      model_name = model.__class__.__name__

      train_size = src_train_size+tgt_train_size
      optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
      fitting_outputs = train_epoch_dann(src_train_loader, tgt_train_loader, src_train_size, tgt_train_size, device, 
                                          model, 
                                          class_criterion, domain_criterion, optimizer, epoch)
      
      train_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc = fitting_outputs
      # print('Epoch {}(train)\t\tloss_avg\tsrc_class_loss\ttgt_class_loss\tsrc_dm_loss\ttgt_dm_loss\tsrc_class_acc\ttgt_class_acc\tdm_acc'.format(epoch))
      # print('\t\t\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(train_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc))

      train_loss_avg_epochs[epoch] = train_loss_avg
      train_src_class_acc_epochs[epoch] = src_class_acc
      train_tgt_class_acc_epochs[epoch] = tgt_class_acc
      train_domain_acc[epoch] = domain_acc
      df_performance.loc[i_CV,['train_src_class_loss','train_tgt_class_loss','train_src_domain_loss','train_tgt_domain_loss', 
                                'train_src_class_acc','train_tgt_class_acc','train_domain_acc']] = [src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc]

      val_outputs = val_epoch_dann(src_val_loader, tgt_val_loader, src_val_size, tgt_val_size, device, 
                                      model,
                                      class_criterion, domain_criterion, epoch)

      val_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc = val_outputs
      # print('Epoch {}(val)\t\tloss_avg\tsrc_class_loss\ttgt_class_loss\tsrc_dm_loss\ttgt_dm_loss\tsrc_class_acc\ttgt_class_acc\tdm_acc'.format(epoch))
      # print('\t\t\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}'.format(val_loss_avg, src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc))

      val_loss_avg_epochs[epoch] = val_loss_avg
      val_src_class_acc_epochs[epoch] = src_class_acc
      val_tgt_class_acc_epochs[epoch] = tgt_class_acc
      val_domain_acc[epoch] = domain_acc

      # 4. store the performance of the model at the last epoch
      df_performance.loc[i_CV,['val_src_class_loss','val_tgt_class_loss','val_src_domain_loss','val_tgt_domain_loss', 
                                'val_src_class_acc','val_tgt_class_acc','val_domain_acc']] = [src_class_loss_avg, tgt_class_loss_avg, src_domain_loss_avg, tgt_domain_loss_avg, src_class_acc, tgt_class_acc, domain_acc]
    
    fig = plt.figure(figsize=(20, 3), dpi=80)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title('loss_avg_epochs')
    ax1.set_xlabel('epoch')
    ax1.plot(np.arange(num_epochs), train_loss_avg_epochs, color='blue', label='train')
    ax1.plot(np.arange(num_epochs), val_loss_avg_epochs, color='red', label='val')
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title('src_class_acc_epochs')
    ax2.set_xlabel('epoch')
    ax2.plot(np.arange(num_epochs), train_src_class_acc_epochs, color='blue', label='train')
    ax2.plot(np.arange(num_epochs), val_src_class_acc_epochs, color='red', label='val')
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_title('tgt_class_acc_epochs')
    ax3.set_xlabel('epoch')
    ax3.plot(np.arange(num_epochs), train_tgt_class_acc_epochs, color='blue', label='train')
    ax3.plot(np.arange(num_epochs), val_tgt_class_acc_epochs, color='red', label='val')
    ax3.legend(loc="upper right")

    # print(val_tgt_class_acc_epochs)

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title('domain_acc')
    ax4.set_xlabel('epoch')
    ax4.plot(np.arange(num_epochs), train_domain_acc, color='blue', label='train')
    ax4.plot(np.arange(num_epochs), val_domain_acc, color='red', label='val')
    ax4.legend(loc="upper right")

    plt.show()

    
    print('=================Exporting pytorch model=================')
    loaded_model = DannModel(device, class_N=classes_n, domain_N=2, channel_n=channel_n, input_dim=src_input_dim).to(device).float()
    export_model(model, loaded_model, outputdir+'model_CV{}'.format(i_CV))
    print('=========================================================')

  # outputdir = '/content/drive/My Drive/中研院/data_mic/stage2_modeloutput/{}/{}/'.format(dataset_name, sensor_loc)


  # 5. export model performance as df
  print('===============Exporting model performance===============')
  export_perofmance(df_performance, CV_n, outputdir)

  print('val_src_class_acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_src_class_acc'], df_performance.loc['std']['val_src_class_acc']))
  print('val_tgt_class_acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_tgt_class_acc'], df_performance.loc['std']['val_tgt_class_acc']))
  print('val_domain_acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_domain_acc'], df_performance.loc['std']['val_domain_acc']))

  print('=========================================================')

  # 6. export notebook parameters as dict
  # datetime object containing current date and time
  print('==============Exporting notebook parameters==============')
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
      'batch_size': batch_size,
      'input_dim': (batch_size, src_train_loader.dataset.data.size()[1], src_train_loader.dataset.data.size()[2]),
      'output_dim': 2,
      'label_dim': src_train_loader.dataset.labels[0:batch_size].data.detach().cpu().numpy().shape,
  }
  print(param_dict)

  with open(outputdir+'notebook_param.json', 'w') as fp:
    json.dump(param_dict, fp)
  print('=========================================================')


  print('val_tgt_class_acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_tgt_class_acc'], df_performance.loc['std']['val_tgt_class_acc']))
  print('val_domain_acc: {:.4f}±{:.4f}'.format(df_performance.loc['mean']['val_domain_acc'], df_performance.loc['std']['val_domain_acc']))

  return (df_performance.loc['mean']['val_tgt_class_acc'], df_performance.loc['std']['val_tgt_class_acc']), (df_performance.loc['mean']['val_domain_acc'], df_performance.loc['std']['val_domain_acc'])
