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
from training_util import *

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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
<<<<<<< HEAD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

dpi = 80

def baseline_learning_diagnosis(num_epochs, train_loss_avg_epochs, val_src_loss_avg_epochs, val_tgt_loss_avg_epochs, train_class_acc_epochs, val_src_class_acc_epochs, val_tgt_class_acc_epochs, i_CV, outputdir):
  if not os.path.exists(outputdir):
      os.makedirs(outputdir)
  print('outputdir for baseline_learning_diagnosis output:', outputdir)
  fig = plt.figure(figsize=(10, 3), dpi=dpi)
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
  fig.savefig(outputdir+'learning_curve_CV{}'.format(i_CV))

def dann_learning_diagnosis(num_epochs, train_loss_avg_epochs, val_loss_avg_epochs, \
train_src_class_acc_epochs, val_src_class_acc_epochs, \
train_tgt_class_acc_epochs, val_tgt_class_acc_epochs, \
train_domain_acc, val_domain_acc, i_CV, outputdir):

  if not os.path.exists(outputdir):
      os.makedirs(outputdir)
  print('outputdir for dann_learning_diagnosis output:', outputdir)

  fig = plt.figure(figsize=(20, 3), dpi=dpi)
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

  ax4 = fig.add_subplot(1, 4, 4)
  ax4.set_title('domain_acc')
  ax4.set_xlabel('epoch')
  ax4.plot(np.arange(num_epochs), train_domain_acc, color='blue', label='train')
  ax4.plot(np.arange(num_epochs), val_domain_acc, color='red', label='val')
  ax4.legend(loc="upper right")

  plt.show()
  fig.savefig(outputdir+'learning_curve_CV{}'.format(i_CV))


def model_output_diagnosis(model, src_loader, tgt_loader, device, fig, col_name, ax_idx):
=======
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def model_output_diagnosis(model, src_loader, tgt_loader, param_dict, training_params, device, fig, col_name, ax_idx):
>>>>>>> 13252fce46b87f1c9c9f8b01ca714d9b2f501eda
  src_data = src_loader.dataset.data
  src_labels = src_loader.dataset.labels
  tgt_data = tgt_loader.dataset.data
  tgt_labels = tgt_loader.dataset.labels

  src_data = src_data.to(device)
  src_labels = src_labels.to(device).long()
  tgt_data = tgt_data.to(device)
  tgt_labels = tgt_labels.to(device).long()

  src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
  tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

  src_feature, src_class_out, src_domain_out = model(src_data)
  tgt_feature, tgt_class_out, tgt_domain_out = model(tgt_data)

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
  data_size = src_class_pred.shape[0]
  (src_class_pred==src_labels.data.detach().cpu().numpy()).sum()/data_size

  src_domain_labels = np.zeros(src_domain_pred.shape[0])
  tgt_domain_labels = np.ones(tgt_domain_pred.shape[0])

  src_class_acc = (src_class_pred==src_labels.data.detach().cpu().numpy()).sum()/data_size
  src_domain_acc = (src_domain_pred==src_domain_labels).sum()/data_size
  tgt_class_acc = (tgt_class_pred==tgt_labels.data.detach().cpu().numpy()).sum()/data_size
  tgt_domain_acc = (tgt_domain_pred==tgt_domain_labels).sum()/data_size
  print('acc performance:', src_class_acc, src_domain_acc, tgt_class_acc, tgt_domain_acc)

  # fig = plt.figure(figsize=(10, 10), dpi=100)

  ax1 = fig.add_subplot(4, 2, ax_idx[0])
  ax1.plot(src_class_sigmoid[:,1],'.b', label='src_class_sigmoid', markersize=3)
  ax1.plot(src_labels.data.detach().cpu().numpy(),'r', alpha=0.5, label='src_class_labels')
  # ax1.set_title('src_class_sigmoid (adl=0, fall=1)')
  ax1.legend(loc='upper right')
  ax1.set_title(col_name, fontsize=20)
  # ax1.set_ylabel('src_class_sigmoid (adl=0, fall=1)', rotation=0, size='large')

  ax2 = fig.add_subplot(4, 2, ax_idx[1])
  ax2.plot(src_domain_sigmoid[:,0],'.b', label='src_domain_sigmoid', markersize=3)
  ax2.plot(src_domain_labels,'r', alpha=0.5, label='src_domain_labels')
  # ax2.set_title('src_domain_sigmoid (src=0, tgt=1)')
  ax2.legend(loc='upper right')

  ax3 = fig.add_subplot(4, 2, ax_idx[2])
  ax3.plot(tgt_class_sigmoid[:,1],'.b', label='tgt_class_sigmoid', markersize=3)
  ax3.plot(tgt_labels.data.detach().cpu().numpy(),'r', alpha=0.5, label='tgt_class_labels')
  # ax3.set_title('tgt_class_sigmoid (adl=0, fall=1)')
  ax3.legend(loc='upper right')

  ax4 = fig.add_subplot(4, 2, ax_idx[3])
  ax4.plot(tgt_domain_sigmoid[:,0],'.b', label='tgt_domain_sigmoid', markersize=3)
  ax4.plot(tgt_domain_labels,'r', alpha=0.5, label='tgt_domain_labels')
  # ax4.set_title('tgt_domain_sigmoid (src=0, tgt=1)')
  ax4.legend(loc='upper right')

<<<<<<< HEAD
def model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, plt_title, i_CV, outputdir):
    if not os.path.exists(outputdir):
      os.makedirs(outputdir)
    print('outputdir for model_output_diagnosis_trainval output:', outputdir)
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    model_output_diagnosis(model, src_train_loader, tgt_train_loader, device, fig, 'train'+plt_title, ax_idx=[1,3,5,7])
    model_output_diagnosis(model, src_val_loader, tgt_val_loader, device, fig, 'val'+plt_title, ax_idx=[2,4,6,8])
    ax_list = fig.axes
    ax_list[0].set_ylabel('src_class', size='large')
    ax_list[1].set_ylabel('src_domain', size='large')
    ax_list[2].set_ylabel('tgt_class', size='large')
    ax_list[3].set_ylabel('tgt_domain', size='large')
    fig.tight_layout()
    plt.show()
    fig.savefig(outputdir+'class_out_diagnosis_CV{}{}'.format(i_CV, plt_title))


def model_features_diagnosis(model, src_loader, tgt_loader, device, ax, col_name):
=======
  # plt.show()

def model_features_diagnosis(model, src_loader, tgt_loader, device):
>>>>>>> 13252fce46b87f1c9c9f8b01ca714d9b2f501eda
  src_data = src_loader.dataset.data
  src_labels = src_loader.dataset.labels
  tgt_data = tgt_loader.dataset.data
  tgt_labels = tgt_loader.dataset.labels

  src_data = src_data.to(device)
  src_labels = src_labels.to(device).long()
  tgt_data = tgt_data.to(device)
  tgt_labels = tgt_labels.to(device).long()

  src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
  tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

  src_feature, src_class_out, src_domain_out = model(src_data)
  tgt_feature, tgt_class_out, tgt_domain_out = model(tgt_data)


  feature_np = torch.cat([src_feature, tgt_feature], dim=0).data.detach().cpu().numpy()
  labels_np = torch.cat([src_labels, tgt_labels], dim=0).data.detach().cpu().numpy()

  domain_np = np.concatenate((src_domain_labels.data.detach().cpu().numpy(), tgt_domain_labels.data.detach().cpu().numpy()), axis=0)

  feature_np = StandardScaler().fit_transform(feature_np) # normalizing the features
  print('show standardize mean and std:', np.mean(feature_np),np.std(feature_np))

<<<<<<< HEAD
  pca_features = PCA(n_components=10)
  principalComponents_features = pca_features.fit_transform(feature_np)
  # variance = pca_features.explained_variance_ratio_ #calculate variance ratios
  var_pca = np.cumsum(np.round(pca_features.explained_variance_ratio_, decimals=3)*100)
  print('PCA var:', var_pca)
  explained_var = var_pca[1]

  # lda = LinearDiscriminantAnalysis(n_components=1)
  # principalComponents_features = lda.fit_transform(feature_np, labels_np)
  # print(feature_np.shape, labels_np.shape, principalComponents_features.shape)
  # # # X_lda = lda.fit(feature_np, labels_np).transform(feature_np)
  # var_lda = lda.explained_variance_ratio_
  # print('LDA var:', var_lda)

  # sys.exit()

  # fig = plt.figure(figsize=(7, 7), dpi=100)
  # ax = fig.add_subplot(111)
  ax.set_xlabel('Principal Component - 1',fontsize=12)
  ax.set_ylabel('Principal Component - 2',fontsize=12)
  ax.set_title('{} (explained_var: {:.2f}%)'.format(col_name, explained_var),fontsize=15)
  # ax.set_title('PCA of features extracted by Gf ({})'.format(col_name),fontsize=15)
  ax.tick_params(axis='both', which='major', labelsize=12)

  class_ids = [0, 1] # adl, fall
  domain_ids = [0, 1] # src, tgt
  colors = ['r', 'g']
  # markers = ['_', 'o']
  markers = ['o', 'x']
=======
  pca_features = PCA(n_components=3)
  principalComponents_features = pca_features.fit_transform(feature_np)

  fig = plt.figure(figsize=(7, 7), dpi=100)
  ax = fig.add_subplot(111)
  ax.set_xlabel('Principal Component - 1',fontsize=20)
  ax.set_ylabel('Principal Component - 2',fontsize=20)
  ax.set_title("PCA of features extracted by Gf",fontsize=20)
  ax.tick_params(axis='both', which='major', labelsize=12)

  class_ids = [0, 1]
  domain_ids = [0, 1]
  colors = ['r', 'g']
  markers = ['x', '.']
>>>>>>> 13252fce46b87f1c9c9f8b01ca714d9b2f501eda
  legend_dict = {
      '00': 'adl_src',
      '01': 'adl_tgt',
      '10': 'fall_src',
      '11': 'fall_tgt',
  }

  pt_label = ['']

  for class_id, marker in zip(class_ids,markers):
    for domain_id, color in zip(domain_ids,colors):
<<<<<<< HEAD
      indicesToKeep = np.where((labels_np==class_id) & (domain_np==domain_id))[0]

      if class_id == 1:
        alpha = 0.3
        ax.scatter(principalComponents_features[indicesToKeep, 0], 
                    principalComponents_features[indicesToKeep, 1], 
                    s = 50, marker=marker, c=color, alpha=alpha,
                  label=legend_dict[str(class_id)+str(domain_id)])
      else:
        alpha = 0.3
        ax.scatter(principalComponents_features[indicesToKeep, 0], 
                    principalComponents_features[indicesToKeep, 1], 
                    s = 50, marker=marker, edgecolors=color, facecolors='None', alpha=alpha,
                  label=legend_dict[str(class_id)+str(domain_id)])

      # ax.scatter(principalComponents_features[indicesToKeep, 0], 
      #            principalComponents_features[indicesToKeep, 1],
      #            c = color, s = 50, marker=marker, alpha=alpha,
      #            label=legend_dict[str(class_id)+str(domain_id)])

  ax.legend(loc='upper right', prop={'size': 15})


def model_features_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, plt_title, i_CV, outputdir):
    if not os.path.exists(outputdir):
      os.makedirs(outputdir)
    print('outputdir for model_features_diagnosis_trainval output:', outputdir)
    fig = plt.figure(figsize=(13, 5), dpi=dpi)
    fig.suptitle('PCA of features extracted by Gf', fontsize=18)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    model_features_diagnosis(model, src_train_loader, tgt_train_loader, device, ax1, 'train'+plt_title)
    model_features_diagnosis(model, src_val_loader, tgt_val_loader, device, ax2, 'val'+plt_title)
    # fig.tight_layout()
    plt.show()
    fig.savefig(outputdir+'feature_diagnosis_CV{}{}'.format(i_CV, plt_title))


# def model_features_diagnosis_3d(model, src_loader, tgt_loader, device, ax, col_name):

#   src_data = src_loader.dataset.data
#   src_labels = src_loader.dataset.labels
#   tgt_data = tgt_loader.dataset.data
#   tgt_labels = tgt_loader.dataset.labels

#   src_data = src_data.to(device)
#   src_labels = src_labels.to(device).long()
#   tgt_data = tgt_data.to(device)
#   tgt_labels = tgt_labels.to(device).long()

#   src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
#   tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

#   src_feature, src_class_out, src_domain_out = model(src_data)
#   tgt_feature, tgt_class_out, tgt_domain_out = model(tgt_data)

#   feature_np = torch.cat([src_feature, tgt_feature], dim=0).data.detach().cpu().numpy()
#   labels_np = torch.cat([src_labels, tgt_labels], dim=0).data.detach().cpu().numpy()
#   domain_np = np.concatenate((src_domain_labels.data.detach().cpu().numpy(), tgt_domain_labels.data.detach().cpu().numpy()), axis=0)

#   feature_np = StandardScaler().fit_transform(feature_np) # normalizing the features
#   print('show standardize mean and std:', np.mean(feature_np),np.std(feature_np))

#   pca_features = PCA(n_components=3)
#   principalComponents_features = pca_features.fit_transform(feature_np)

#   class_ids = [0, 1]
#   domain_ids = [0, 1]
#   colors = ['r', 'g']
#   markers = ['_', '+']
#   legend_dict = {
#       '00': 'adl_src',
#       '01': 'adl_tgt',
#       '10': 'fall_src',
#       '11': 'fall_tgt',
#   }

#   pt_label = ['']

#   # fig = plt.figure(figsize=(10, 10), dpi=100)
#   # ax = fig.add_subplot(111, projection='3d')
#   ax.set_xlabel('Principal Component - 1',fontsize=20)
#   ax.set_ylabel('Principal Component - 2',fontsize=20)
#   ax.set_zlabel('Principal Component - 3',fontsize=20)
#   ax.set_title('PCA of features extracted by Gf ({})'.format(col_name),fontsize=20)
#   markersize = 50
#   for class_id, marker in zip(class_ids,markers):
#     for domain_id, color in zip(domain_ids,colors):
#       indicesToKeep = np.where((labels_np==class_id) & (domain_np==domain_id))[0]
#       # if domain_id == 1:
#       #   markersize = 0
#       # else:
#       #   markersize = 50
#       ax.scatter(principalComponents_features[indicesToKeep, 0], 
#                  principalComponents_features[indicesToKeep, 1],
#                  principalComponents_features[indicesToKeep, 2],
#                  c = color, s = markersize, marker=marker,
#                  label=legend_dict[str(class_id)+str(domain_id)])

#   ax.legend(loc='upper right', prop={'size': 15})

def get_mean(mean_std):
  return float(mean_std.split('±')[0])

def get_std(mean_std):
  return float(mean_std.split('±')[1])

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)

def get_rep10_stats(inputdir, rep_n):
  df_performance_table_all = pd.DataFrame('', index=['source', 'DANN', 'target', 'domain', 'time_elapsed'], columns=[])
  for i in range(rep_n):
    df_inputdir = inputdir+'df_performance_table_rep10_i{}.csv'.format(i)
    df_performance_table = pd.read_csv(df_inputdir, index_col=0)
    df_performance_table = df_performance_table.loc[ ['source', 'DANN', 'target', 'domain', 'time_elapsed'] , :]
    df_performance_table_all = pd.concat([df_performance_table_all, df_performance_table], axis=1)
    df_performance_table_all_mean = df_performance_table_all.applymap(get_mean)

  df_performance_table_means = df_performance_table_all_mean.mean(axis=1)
  df_performance_table_stds = df_performance_table_all_mean.std(axis=1)

  df_performance_table_all['mean'] = df_performance_table_means
  df_performance_table_all['std'] = df_performance_table_stds
  df_performance_table_all['rep10'] = df_performance_table_all[['mean', 'std']].apply(lambda x : '{:.3f}±{:.3f}'.format(x[0],x[1]), axis=1)

  print('df_performance_table_all saved at', inputdir)
  df_performance_table_all.to_csv(inputdir+'df_performance_table_agg.csv', encoding='utf-8')

  return df_performance_table_all


=======
      # if class_id == 1:
      #   markersize = 0
      # else:
      #   markersize = 1
      indicesToKeep = np.where((labels_np==class_id) & (domain_np==domain_id))[0]
      ax.scatter(principalComponents_features[indicesToKeep, 0], 
                  principalComponents_features[indicesToKeep, 1],
                  c = color, s = 50, marker=marker, label=legend_dict[str(class_id)+str(domain_id)])

  ax.legend(prop={'size': 15})

  plt.show()

def model_features_diagnosis_3d(model, src_loader, tgt_loader, device):

  src_data = src_loader.dataset.data
  src_labels = src_loader.dataset.labels
  tgt_data = tgt_loader.dataset.data
  tgt_labels = tgt_loader.dataset.labels

  src_data = src_data.to(device)
  src_labels = src_labels.to(device).long()
  tgt_data = tgt_data.to(device)
  tgt_labels = tgt_labels.to(device).long()

  src_domain_labels = torch.zeros(src_data.size()[0]).to(device).long()
  tgt_domain_labels = torch.ones(tgt_data.size()[0]).to(device).long()

  src_feature, src_class_out, src_domain_out = model(src_data)
  tgt_feature, tgt_class_out, tgt_domain_out = model(tgt_data)

  feature_np = torch.cat([src_feature, tgt_feature], dim=0).data.detach().cpu().numpy()
  labels_np = torch.cat([src_labels, tgt_labels], dim=0).data.detach().cpu().numpy()
  domain_np = np.concatenate((src_domain_labels.data.detach().cpu().numpy(), tgt_domain_labels.data.detach().cpu().numpy()), axis=0)

  feature_np = StandardScaler().fit_transform(feature_np) # normalizing the features
  print('show standardize mean and std:', np.mean(feature_np),np.std(feature_np))

  pca_features = PCA(n_components=3)
  principalComponents_features = pca_features.fit_transform(feature_np)

  class_ids = [0, 1]
  domain_ids = [0, 1]
  colors = ['r', 'g']
  markers = ['x', '.']
  legend_dict = {
      '00': 'adl_src',
      '01': 'adl_tgt',
      '10': 'fall_src',
      '11': 'fall_tgt',
  }

  pt_label = ['']

  fig = plt.figure(figsize=(10, 10), dpi=100)
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel('Principal Component - 1',fontsize=20)
  ax.set_ylabel('Principal Component - 2',fontsize=20)
  ax.set_zlabel('Principal Component - 3',fontsize=20)
  ax.set_title("PCA of features extracted by Gf",fontsize=20)
  markersize = 50
  for class_id, marker in zip(class_ids,markers):
    for domain_id, color in zip(domain_ids,colors):
      indicesToKeep = np.where((labels_np==class_id) & (domain_np==domain_id))[0]
      # if domain_id == 1:
      #   markersize = 0
      # else:
      #   markersize = 50
      ax.scatter(principalComponents_features[indicesToKeep, 0], 
                 principalComponents_features[indicesToKeep, 1],
                 principalComponents_features[indicesToKeep, 2],
                 c = color, s = markersize, marker=marker,
                 label=legend_dict[str(class_id)+str(domain_id)])

  ax.legend(prop={'size': 15})
>>>>>>> 13252fce46b87f1c9c9f8b01ca714d9b2f501eda
