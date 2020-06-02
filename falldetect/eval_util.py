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
from falldetect.training_util import *

import time
import datetime
from datetime import datetime
import json

# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc( 'savefig', facecolor = 'white' )
# matplotlib.rc( 'savefig', transparent=True )

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

dpi = 80

# def baseline_learning_diagnosis(num_epochs, train_performance_dict_list, val_src_performance_dict_list, val_tgt_performance_dict_list, PAD_list, i_CV, outputdir):
#   train_performance_epochs = pd.DataFrame(train_performance_dict_list)
#   val_src_performance_epochs = pd.DataFrame(val_src_performance_dict_list)
#   val_tgt_performance_epochs = pd.DataFrame(val_tgt_performance_dict_list)

# #   metric_list = ['loss', 'acc', 'sensitivity', 'precision', 'F1']
#   metric_list = ['loss', 'acc', 'sensitivity', 'precision', 'F1', 'PAD']

#   if not os.path.exists(outputdir):
#       os.makedirs(outputdir)
#   print('outputdir for baseline_learning_diagnosis output:', outputdir)
#   fig = plt.figure(figsize=(5*len(metric_list), 3), dpi=dpi)

#   for i, metric_name in enumerate(metric_list):
#     ax1 = fig.add_subplot(1, len(metric_list), i+1)
#     ax1.set_title('{}_epochs'.format(metric_name))
#     ax1.set_xlabel('epoch')
#     if metric_name=='PAD':
#       ax1.plot(np.arange(num_epochs), PAD_list, color='blue', label='PAD_val')
#     else:
#       ax1.plot(np.arange(num_epochs), train_performance_epochs['{}'.format(metric_name)].values, color='blue', label='train')
#       ax1.plot(np.arange(num_epochs), val_src_performance_epochs['src_{}'.format(metric_name)].values, color='red', label='val_src')
#       ax1.plot(np.arange(num_epochs), val_tgt_performance_epochs['tgt_{}'.format(metric_name)].values, color='green', label='val_tgt')
#     ax1.legend(loc="upper right")
# #     plt.show()
#     fig.savefig(outputdir+'learning_curve_CV{}'.format(i_CV))

# def dann_learning_diagnosis(num_epochs, train_performance_dict_list, val_performance_dict_list, PAD_list, i_CV, outputdir):
def dann_learning_diagnosis(num_epochs, train_performance_dict_list, val_performance_dict_list, PAD_list, i_CV, epoch_optimal, metric_list, outputdir):
  train_performance_epochs = pd.DataFrame(train_performance_dict_list)
  val_performance_epochs = pd.DataFrame(val_performance_dict_list)

  if not os.path.exists(outputdir):
      os.makedirs(outputdir)
  print('outputdir for dann_learning_diagnosis output:', outputdir)

#   metric_list = ['src_class_loss', 'src_class_acc', 'tgt_class_acc', 'tgt_sensitivity', 'tgt_precision', 'tgt_F1', 'domain_acc']
#   metric_list = ['src_class_loss', 'src_acc', 'tgt_acc', 'tgt_sensitivity', 'tgt_precision', 'tgt_F1', 'domain_acc', 'PAD']


  fig = plt.figure(figsize=(5*len(metric_list), 3), dpi=dpi)

  for i, metric_name in enumerate(metric_list):
      ax1 = fig.add_subplot(1, len(metric_list), i+1)
      ax1.set_title('{}_epochs'.format(metric_name))
      ax1.set_xlabel('epoch')
      if metric_name=='PAD':
        ax1.plot(np.arange(num_epochs), PAD_list, color='blue', label='PAD_val')
      elif metric_name=='total_loss':
        ax1.plot(np.arange(num_epochs), train_performance_epochs['{}'.format(metric_name)].values, color='blue', label='train')
        ax1.plot(np.arange(num_epochs), val_performance_epochs['{}'.format(metric_name)].values, color='red', label='val')
      else:
        ax1.plot(np.arange(num_epochs), train_performance_epochs['src_{}'.format(metric_name)].values, color='blue', label='train')
        ax1.plot(np.arange(num_epochs), val_performance_epochs['src_{}'.format(metric_name)].values, color='red', label='val_src')
        ax1.plot(np.arange(num_epochs), val_performance_epochs['tgt_{}'.format(metric_name)].values, color='green', label='val_tgt')

      ax1.axvline(epoch_optimal, linestyle='--', color='gray', alpha=0.7, label='checkpoint')
      ax1.legend(loc="upper right")
		
  fig.savefig(outputdir+'learning_curve_CV{}'.format(i_CV))

def model_output_diagnosis(model, src_loader, tgt_loader, device, fig, col_name, ax_idx):
#   metric_name = 'F1'

  model.eval()
  src_data = src_loader.dataset.data
  src_labels = src_loader.dataset.labels
  src_DataNameList_idx = src_loader.dataset.DataNameList_idx
  tgt_data = tgt_loader.dataset.data
  tgt_labels = tgt_loader.dataset.labels
  tgt_DataNameList_idx = tgt_loader.dataset.DataNameList_idx

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

  src_domain_labels = np.zeros(src_domain_pred.shape[0])
  tgt_domain_labels = np.ones(tgt_domain_pred.shape[0])

  src_class_acc = (src_class_pred==src_labels.data.detach().cpu().numpy()).sum()/data_size
  src_domain_acc = (src_domain_pred==src_domain_labels).sum()/data_size
  tgt_class_acc = (tgt_class_pred==tgt_labels.data.detach().cpu().numpy()).sum()/data_size
  tgt_domain_acc = (tgt_domain_pred==tgt_domain_labels).sum()/data_size

  src_class_F1 = f1_score(src_labels.data.detach().cpu().numpy(), src_class_pred, zero_division=1)
#   src_domain_F1 = f1_score(src_domain_labels, src_domain_pred, zero_division=1)
  tgt_class_F1 = f1_score(tgt_labels.data.detach().cpu().numpy(), tgt_class_pred, zero_division=1)
#   tgt_domain_F1 = f1_score(tgt_domain_labels, tgt_domain_pred, zero_division=1)
  

  ax1 = fig.add_subplot(4, 2, ax_idx[0])
  ax1.plot(src_class_sigmoid[:,1],'.b', label='src_class_sigmoid', markersize=3)
  ax1.plot(src_class_pred,'b', alpha=0.5, label='src_class_decision')
  ax1.plot(src_labels.data.detach().cpu().numpy(),'r', alpha=0.5, label='src_class_labels')
  # ax1.set_title('src_class_sigmoid (adl=0, fall=1)')
  ax1.legend(loc='upper right')
  ax1.set_title(col_name+'\n(src/tgt F1={:.2f}/{:.2f})'.format(src_class_F1, tgt_class_F1), fontsize=15)

  ax2 = fig.add_subplot(4, 2, ax_idx[1])
  ax2.plot(src_domain_sigmoid[:,0],'.b', label='src_domain_sigmoid', markersize=3)
  ax2.plot(src_domain_labels,'r', alpha=0.5, label='src_domain_labels')
  ax2.legend(loc='upper right')

  ax3 = fig.add_subplot(4, 2, ax_idx[2])
  ax3.plot(tgt_class_sigmoid[:,1],'.b', label='tgt_class_sigmoid', markersize=3)
  ax3.plot(tgt_class_pred,'b', alpha=0.5, label='tgt_class_decision')
  ax3.plot(tgt_labels.data.detach().cpu().numpy(),'r', alpha=0.5, label='tgt_class_labels')
  ax3.legend(loc='upper right')

  ax4 = fig.add_subplot(4, 2, ax_idx[3])
  ax4.plot(tgt_domain_sigmoid[:,0],'.b', label='tgt_domain_sigmoid', markersize=3)
  ax4.plot(tgt_domain_labels,'r', alpha=0.5, label='tgt_domain_labels')
  ax4.legend(loc='upper right')
	
	
  return np.stack((src_class_sigmoid[:,1], src_DataNameList_idx), axis=1), np.stack((tgt_class_sigmoid[:,1], tgt_DataNameList_idx), axis=1)

#   return src_class_sigmoid[:,1], tgt_class_sigmoid[:,1]

def model_output_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, plt_title, i_CV, outputdir):
    model.eval()
    if not os.path.exists(outputdir):
      os.makedirs(outputdir)
    print('outputdir for model_output_diagnosis_trainval output:', outputdir)
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    src_class_sigmoid, tgt_class_sigmoid = model_output_diagnosis(model, src_train_loader, tgt_train_loader, device, fig, 'train'+plt_title, ax_idx=[1,3,5,7])
    src_class_sigmoid, tgt_class_sigmoid = model_output_diagnosis(model, src_val_loader, tgt_val_loader, device, fig, 'val'+plt_title, ax_idx=[2,4,6,8])
    ax_list = fig.axes
    ax_list[0].set_ylabel('src_class', size='large')
    ax_list[1].set_ylabel('src_domain', size='large')
    ax_list[2].set_ylabel('tgt_class', size='large')
    ax_list[3].set_ylabel('tgt_domain', size='large')
    fig.tight_layout()
    plt.show()
    fig.savefig(outputdir+'class_out_diagnosis_CV{}{}'.format(i_CV, plt_title))
	
    data_saver(src_class_sigmoid, 'src_class_sigmoid_CV{}'.format(i_CV), outputdir)
    data_saver(tgt_class_sigmoid, 'tgt_class_sigmoid_CV{}'.format(i_CV), outputdir)
    
#     if aaa == None:
#       pass
#     else:
#     plt.plot(aaa, alpha=0.3)
#     plt.plot(src_train_loader.dataset.labels.data.numpy()*0.5, 'r', alpha=0.3)
#     plt.show()
#     print('diff:', (aaa-src_train_loader.dataset.labels.data.numpy()).sum())
#     sys.exit()
    
def model_features_diagnosis(model, src_loader, tgt_loader, device, ax, col_name):
  model.eval()
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

  pca_features = PCA(n_components=10)
  principalComponents_features = pca_features.fit_transform(feature_np)
  var_pca = np.cumsum(np.round(pca_features.explained_variance_ratio_, decimals=3)*100)
  print('PCA var:', var_pca)
  explained_var = var_pca[1]

  ax.set_xlabel('Principal Component - 1',fontsize=12)
  ax.set_ylabel('Principal Component - 2',fontsize=12)
  ax.set_title('{} (explained_var: {:.2f}%)'.format(col_name, explained_var),fontsize=15)
  # ax.set_title('PCA of features extracted by Gf ({})'.format(col_name),fontsize=15)
  ax.tick_params(axis='both', which='major', labelsize=12)

  class_ids = [0, 1] # adl, fall
  domain_ids = [0, 1] # src, tgt
  colors = ['r', 'g']
  markers = ['o', 'x']
  legend_dict = {
      '00': 'adl_src',
      '01': 'adl_tgt',
      '10': 'fall_src',
      '11': 'fall_tgt',
  }

  pt_label = ['']

  for class_id, marker in zip(class_ids,markers):
    for domain_id, color in zip(domain_ids,colors):
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

  ax.legend(loc='upper right', prop={'size': 15})


def model_features_diagnosis_trainval(model, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, device, plt_title, i_CV, outputdir):
    model.eval()
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


def get_mean(mean_std):
  return float(mean_std.split('±')[0])

def get_std(mean_std):
  return float(mean_std.split('±')[1])

def get_PAD(src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, model, device, c=3000):
#         start_time = time.time()

	model.eval()

	data = src_train_loader.dataset.data.to(device)
	src_domain_labels = np.zeros(data.shape[0])
	src_feature_out, _, _ = model(data)

	data = tgt_train_loader.dataset.data.to(device)
	tgt_domain_labels = np.ones(data.shape[0])
	tgt_feature_out, _, _ = model(data)

	train_data = np.concatenate((src_feature_out.data.detach().cpu().numpy(),tgt_feature_out.data.detach().cpu().numpy()),axis=0)
	train_label = np.concatenate((src_domain_labels,tgt_domain_labels))

# 	print(train_data.shape, train_label.shape)

	svm_model = svm.SVC(C=c, probability=True)
	svm_model.fit(train_data, train_label)

	data = src_val_loader.dataset.data.to(device)
	src_domain_labels = np.zeros(data.shape[0])
	src_feature_out, _, _ = model(data)

	data = tgt_val_loader.dataset.data.to(device)
	tgt_domain_labels = np.ones(data.shape[0])
	tgt_feature_out, _, _ = model(data)

	val_data = np.concatenate((src_feature_out.data.detach().cpu().numpy(),tgt_feature_out.data.detach().cpu().numpy()),axis=0)
	val_label = np.concatenate((src_domain_labels,tgt_domain_labels))

	svm_out = svm_model.predict_proba(val_data)
	mse = mean_squared_error(val_label, svm_out[:,1])
	PAD = 2. * (1. - 2. * mse)
# 	print('\nmse=', mse)
# 	print('PAD=', PAD)

#         time_elapsed = time.time() - start_time
#         print('time elapsed:', time.strftime("%H:%M:%S", time.gmtime(time_elapsed)))

#         sys.exit()

	return PAD

def modification_date(filename):
    t = os.path.getmtime(filename)
    return datetime.datetime.fromtimestamp(t)


def get_rep_stats(df_performance_table_agg, rep_n):
	df_acc = df_performance_table_agg.loc[ ['source', 'DANN', 'target', 'domain', 'PAD_source', 'PAD_DANN'] , ].copy()
	df_params = df_performance_table_agg.loc[ ['channel_n', 'batch_size', 'learning_rate', 'time_elapsed', 'num_params'], ].copy()

    # accs
	df_performance_table_all_mean = df_acc.applymap(get_mean)
	df_performance_table_means = df_performance_table_all_mean.mean(axis=1)
	df_performance_table_stds = df_performance_table_all_mean.std(axis=1)
	df_performance_table_all_mean['mean'] = df_performance_table_means
	df_performance_table_all_mean['std'] = df_performance_table_stds
	df_performance_table_all_mean['rep'] = df_performance_table_all_mean[['mean', 'std']].apply(lambda x : '{:.3f}±{:.3f}'.format(x[0],x[1]), axis=1)

    # params
	df_params_means = df_params.mean(axis=1)

	df_performance_table_agg['rep_avg'] = ''
	df_performance_table_agg.loc[ ['source','DANN','target','domain','PAD_source','PAD_DANN'] , ['rep_avg']] = df_performance_table_all_mean.loc[:, 'rep']
	df_performance_table_agg.loc[ ['channel_n','batch_size','learning_rate','time_elapsed','num_params'], ['rep_avg']] = df_params_means
	return df_performance_table_agg


def get_optimal_v0(df_performance_table_agg):
    df_performance_table_agg_temp = df_performance_table_agg.copy()

    result = df_performance_table_agg_temp[['HP_i0','HP_i1','HP_i2']].sort_values(by='DANN', ascending=False, axis=1)
    batch_size_optimal = result.loc['batch_size'][0]

    result = df_performance_table_agg_temp[['HP_i3','HP_i3_1','HP_i4']].sort_values(by='DANN', ascending=False, axis=1)
    channel_n_optimal = result.loc['channel_n'][0]

    result = df_performance_table_agg_temp[['HP_i5','HP_i5_1','HP_i6']].sort_values(by='DANN', ascending=False, axis=1)
    learning_rate_optimal = result.loc['learning_rate'][0]

    return int(batch_size_optimal), int(channel_n_optimal), learning_rate_optimal


def get_optimal_v1(df_performance_table_agg):
    df_performance_table_agg_temp = df_performance_table_agg.copy()

    result = df_performance_table_agg_temp.sort_values(by='DANN', ascending=False, axis=1)
    channel_n_optimal = result.loc['channel_n'][0]
    return int(channel_n_optimal)