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
import math

import random
from collections import Counter
from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_DIM = 50
KERNEL_SIZE = 3

class GradReverse(torch.autograd.Function):
  """
  Extension of grad reverse layer
  """
  @staticmethod
  def forward(ctx, x, constant):
      ctx.constant = constant
      return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
      grad_output = grad_output.neg() * ctx.constant
      return grad_output, None

  def grad_reverse(x, constant):
      return GradReverse.apply(x, constant)



# Convolutional neural network (two convolutional layers)
class FeatureExtractor(nn.Module):
  def __init__(self, input_dim=50, channel_n=16):
      super(FeatureExtractor, self).__init__()
      self.layer1 = nn.Sequential(
          nn.Conv1d(3, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
          nn.BatchNorm1d(channel_n),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      self.layer2 = nn.Sequential(
          nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
          nn.BatchNorm1d(channel_n),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      self.layer3 = nn.Sequential(
          nn.Conv1d(channel_n, channel_n, kernel_size=KERNEL_SIZE, stride=1, padding=2),
          nn.BatchNorm1d(channel_n),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
	
      cnn_layer1_dim = (input_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
      pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

      cnn_layer2_dim = (pool_layer1_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
      pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)
      
      cnn_layer3_dim = (pool_layer2_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
      pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

      self.feature_out_dim = int(pool_layer3_dim*channel_n)
#       self.feature_out_dim = int(pool_layer2_dim*channel_n)
      pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print('FeatureExtractor_total_params:', pytorch_total_params)
      
  def forward(self, x):
    out1 = self.layer1(x.float())
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out3 = out3.reshape(out3.size(0), -1)
    return out3
#     out2 = out2.reshape(out2.size(0), -1)
#     return out2



# # Convolutional neural network (two convolutional layers)
# class FeatureExtractor(nn.Module):
#   def __init__(self, input_dim=50, channel_n=16):
#       super(FeatureExtractor, self).__init__()
#       self.layer1 = nn.Sequential(
#           nn.Conv1d(3, channel_n, kernel_size=3, stride=1, padding=2),
#           nn.BatchNorm1d(channel_n),
#           nn.ReLU(),
#           nn.MaxPool1d(kernel_size=2, stride=2))
#       self.layer2 = nn.Sequential(
#           nn.Conv1d(channel_n, channel_n*2, kernel_size=3, stride=1, padding=2),
#           nn.BatchNorm1d(channel_n*2),
#           nn.ReLU(),
#           nn.MaxPool1d(kernel_size=2, stride=2))
#       # self.layer3 = nn.Sequential(
#       #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2),
#       #     nn.BatchNorm1d(64),
#       #     nn.ReLU(),
#       #     nn.MaxPool1d(kernel_size=2, stride=2))
      
#       cnn_layer1_dim = (input_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#       pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

#       cnn_layer2_dim = (pool_layer1_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#       pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

#       # cnn_layer3_dim = (pool_layer2_dim+2*2-1*(3-1)-1)+1
#       # pool_layer3_dim = (cnn_layer3_dim-1*(2-1)-1)/2+1

#       # print('cnn_layer1_dim:', cnn_layer1_dim)
#       # print('pool_layer1_dim:', pool_layer1_dim)
#       # print('cnn_layer2_dim:', cnn_layer2_dim)
#       # print('pool_layer2_dim:', pool_layer2_dim)
#       # print('cnn_layer3_dim:', cnn_layer3_dim)
#       # print('pool_layer3_dim:', pool_layer3_dim)
#       # fc_dim = int(((((input_dim)+2*2-1)/2+2*2-1)/2+2*2-1)/2*64)
#       # self.fc = nn.Linear(int(pool_layer2_dim)*32, num_classes)
#       self.feature_out_dim = int(pool_layer2_dim*channel_n*2)
#       pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#       print('FeatureExtractor_total_params:', pytorch_total_params)
      
#   def forward(self, x):
#     out1 = self.layer1(x.float())
#     # print('out1 size:', out1.size())
#     out2 = self.layer2(out1)
#     # print('out2 size:', out2.size())
#     # out3 = self.layer3(out2)
#     # print('out3 size:', out3.size())
#     # out3 = out3.reshape(out3.size(0), -1)
#     out2 = out2.reshape(out2.size(0), -1)
#     # print('out2 size:', out2.size())
#     # out3 = self.fc(out2)
#     # print('x, out1, out2, out 3, out4 size',  x.size(), out1.size(), out2.size(), out3.size(), out4.size())
#     return out2


class ClassClassifier(nn.Module):
  def __init__(self, num_classes=10, input_dim=50):
      super(ClassClassifier, self).__init__()
      self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
      self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)
      self.drop = nn.Dropout(p=0.5)
      self.relu = nn.ReLU()
      pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print('ClassClassifier_total_params:', pytorch_total_params)
    
  def forward(self, x):
    out1 = self.relu(self.fc1(x.float()))
    out2 = self.fc2(out1)
    return out2

# domain classifier neural network (fc layers)
class DomainClassifier(nn.Module):
  def __init__(self, num_classes=10, input_dim=50):
      super(DomainClassifier, self).__init__()
      self.fc1 = nn.Linear(input_dim, HIDDEN_DIM)
      self.fc2 = nn.Linear(HIDDEN_DIM, num_classes)
      self.drop = nn.Dropout(p=0.5)
      self.relu = nn.ReLU()
      pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print('DomainClassifier_total_params:', pytorch_total_params)
      
  def forward(self, x, constant):
    out1 = GradReverse.grad_reverse(x.float(), constant)
    out1 = self.relu(self.fc1(out1))
    out2 = self.fc2(out1)
    return out2


class DannModel(nn.Module):
  def __init__(self, device, class_N=2, domain_N=2, channel_n=16, input_dim=10):
    super(DannModel, self).__init__()
    self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n).to(device).float()
	
#     cnn_layer1_dim = (input_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#     pool_layer1_dim = math.floor((cnn_layer1_dim-1*(2-1)-1)/2+1)

#     cnn_layer2_dim = (pool_layer1_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#     pool_layer2_dim = math.floor((cnn_layer2_dim-1*(2-1)-1)/2+1)

#     cnn_layer3_dim = (pool_layer2_dim+2*2-1*(KERNEL_SIZE-1)-1)+1
#     pool_layer3_dim = math.floor((cnn_layer3_dim-1*(2-1)-1)/2+1)

#     feature_out_dim = int(pool_layer3_dim*channel_n)
    feature_out_dim = self.feature_extractor.feature_out_dim
		
    self.class_classfier = ClassClassifier(num_classes=class_N, input_dim=feature_out_dim).to(device).float()
    self.domain_classifier = DomainClassifier(num_classes=domain_N, input_dim=feature_out_dim).to(device).float()

    pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print('DannModel_total_params:', pytorch_total_params)
    

  def forward(self, x):
    feature_out = self.feature_extractor(x)
    class_output = self.class_classfier(feature_out)
    domain_output = self.domain_classifier(feature_out, 1)
    return feature_out, class_output, domain_output


# class BaselineModel(nn.Module):
#   def __init__(self, device, class_N=2, channel_n=16, input_dim=10):
#     super(BaselineModel, self).__init__()
#     self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n).to(device).float()
#     cnn_layer1_dim = (input_dim+2*2-1*(3-1)-1)+1
#     pool_layer1_dim = (cnn_layer1_dim-1*(2-1)-1)/2+1

#     cnn_layer2_dim = (pool_layer1_dim+2*2-1*(3-1)-1)+1
#     pool_layer2_dim = (cnn_layer2_dim-1*(2-1)-1)/2+1

#     feature_out_dim = int(pool_layer2_dim*channel_n*2)

#     self.class_classfier = ClassClassifier(num_classes=class_N, input_dim=feature_out_dim).to(device).float()
#     # self.domain_classifier = DomainClassifier(num_classes=domain_N, input_dim=feature_out_dim).to(device).float()
      
#   def forward(self, x):
#     feature_out = self.feature_extractor(x)
#     # print('feature_out size', feature_out.size())
#     class_out = self.class_classfier(feature_out)
#     # domain_output = self.domain_classifier(feature_out, 1)
#     # return feature_out
#     return feature_out, class_out




# validated, the implementation is correct
def contextExapansion(x, win_size, win_stride, step_n):
  # size of x: torch.Size([batch_size, channel_n, input_size])
  # size of x_seq: torch.Size([step_n, batch_size, channel_n, win_size])

  batch_size = x.size()[0]
  channel_n = x.size()[1]
  input_size = x.size()[2]

  x_seq = torch.ones((step_n, batch_size, channel_n, win_size), dtype=torch.double)
  timesteps = np.asarray(range(win_size))
  for i in range(step_n):
    indices = i*win_stride+timesteps
    x_seq[i, :, :, :] = x[:,:,indices]
  return x_seq

def labelExapansion(y, step_n):
  # size of y: torch.Size([batch_size, 1])
  # size of y_seq: torch.Size([batch_size, step_n])

  batch_size = y.size()[0]

  y_seq = torch.ones((step_n, batch_size), dtype=torch.double)
  timesteps = np.asarray(range(win_size))
  for i in range(step_n):
    y_seq[i, :] = y
  return y_seq


# Convolutional neural network (two convolutional layers)
class CnnLstm(nn.Module):
  def __init__(self, device, class_N=2, channel_n=16, dropout=0.5, hiddenDim_f=5, hiddenDim_y=5, hiddenDim_d=5, win_size=22, win_stride=5, step_n=5):
      super(CnnLstm, self).__init__()
      self.win_size = win_size
      self.win_stride = win_stride
      self.step_n = step_n
      self.device = device
      self.feature_extractor = FeatureExtractor(input_dim=win_size, channel_n=channel_n).to(device).float()
	

      cnn_layer1_dim = (win_size+2*2-1*(3-1)-1)+1
      pool_layer1_dim = (cnn_layer1_dim-1*(2-1)-1)/2+1
      cnn_layer2_dim = (pool_layer1_dim+2*2-1*(3-1)-1)+1
      pool_layer2_dim = (cnn_layer2_dim-1*(2-1)-1)/2+1

      self.feature_out_dim = int(pool_layer2_dim*channel_n*2)

      # # lstm_out_seq size: torch.Size([step_n, batch_size, hiddenDim*2])
      # self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
      #   input_size=self.feature_out_dim,
      #   hidden_size=hiddenDim_f,         # rnn hidden unit
      #   num_layers=2,           # number of rnn layer
      #   batch_first=False,       # input & output will has not batch size as 1st dimension. e.g. (time_step, batch, input_size)
      #   bidirectional=True,
      #   dropout=dropout
      # ).to(device).float()


      # self.class_classifier = ClassClassifier_lstm(num_classes=2, hiddenDim=hiddenDim_y, input_dim=hiddenDim_f*2, steps_n=step_n).to(device).float()
      # self.domain_classifier = DomainClassifier_lstm(num_classes=2, hiddenDim=hiddenDim_d, input_dim=hiddenDim_f*2, steps_n=step_n).to(device).float()
      self.class_classifier = ClassClassifier_lstm(num_classes=2, hiddenDim=hiddenDim_y, input_dim=self.feature_out_dim, steps_n=step_n).to(device).float()
      self.domain_classifier = DomainClassifier_lstm(num_classes=2, hiddenDim=hiddenDim_d, input_dim=self.feature_out_dim, steps_n=step_n).to(device).float()
      
      pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print('CnnLstm_total_params:', pytorch_total_params)
      
  def forward(self, x):
    # size of x: torch.Size([batch_size, channel_n, input_size])
    # size of x_seq: torch.Size([step_n, batch_size, channel_n, win_size])
    debug = False
    x_seq = contextExapansion(x, self.win_size, self.win_stride, self.step_n).to(self.device).float()

    lstm_out_seq = torch.ones((self.step_n, x.size()[0], self.feature_out_dim), dtype=torch.float).to(self.device)

    for t in range(self.step_n):
      # Input: (N, C_in, L_in)
      # Output: (N, L_out=self.feature_out_dim*C_out)
      # print('show size')
      # print(feature_out_seq.size(), x_seq[t,:,:,:].size(), self.feature_extractor(x_seq[t,:,:,:]).size())
      # sys.exit()
      lstm_out_seq[t,:,:] = self.feature_extractor(x_seq[t,:,:,:])
    
    # lstm_out_seq = feature_out_seq
    # Input: (seq_len, batch, input_size)
    # Output: (seq_len, batch, num_directions * hidden_size)
    # lstm_out_seq, (h_n, h_c) = self.lstm(feature_out_seq, None)

    class_output = self.class_classifier(lstm_out_seq)
    domain_output = self.domain_classifier(lstm_out_seq, 1)

    if debug:
      print('CnnLstm')
      print('x size:', x.size())
      print('x_seq size:', x_seq.size())
      print('feature_out_seq size:', feature_out_seq.size())

      print('lstm_out_seq size:', lstm_out_seq.size())
      print('class_output size:', class_output.size())
      print('domain_output size:', domain_output.size())

    lstm_out_seq = lstm_out_seq.transpose(0,1)
    lstm_out_seq = lstm_out_seq.reshape(lstm_out_seq.size()[0],-1)

    return lstm_out_seq, class_output, domain_output



# fall classifier neural network (fc layers)
class ClassClassifier_lstm(nn.Module):
  def __init__(self, num_classes=2, hiddenDim=16, input_dim=50, steps_n=5):
      super(ClassClassifier_lstm, self).__init__()
      self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
        input_size=input_dim,
        hidden_size=hiddenDim,         # rnn hidden unit
        num_layers=1,           # number of rnn layer
        batch_first=False,       # input & output will has not batch size as 1st dimension. e.g. (time_step, time_step, input_size)
        bidirectional=True,
        dropout=0.5
      )
      self.fc1 = nn.Linear(steps_n*hiddenDim*2, 10)
      self.fc2 = nn.Linear(10, num_classes)
      self.relu = nn.ReLU(inplace=False)
      self.fc3 = nn.Linear(steps_n*hiddenDim*2, num_classes)

      # self.lsm = nn.LogSoftmax(dim=1)
      
  def forward(self, x):
    debug = False
    # Input: (seq_len, batch, input_size)
    # Output: (seq_len, batch, num_directions * hidden_size)
    out1_seq, (h_n, h_c) = self.lstm(x)
    out1_seq = out1_seq.transpose(0,1)
    out1_seq = out1_seq.reshape(out1_seq.size()[0],-1)

    # out2 = self.relu(self.fc1(out1_seq))
    # out3 = self.fc2(out2)
    out3 = self.fc3(out1_seq)

    if debug:
      print('ClassClassifier_lstm')
      print('out1_seq size:', out1_seq.size())
      print('out2 size:', out2.size())
      print('out3 size:', out3.size())

    return out3


# domain classifier neural network (fc layers)
class DomainClassifier_lstm(nn.Module):
  def __init__(self, num_classes=2, hiddenDim=16, input_dim=50, steps_n=5):
      super(DomainClassifier_lstm, self).__init__()
      self.lstm = nn.LSTM(         # if use nn.RNN(), it hardly learns
        input_size=input_dim,
        hidden_size=hiddenDim,         # rnn hidden unit
        num_layers=1,           # number of rnn layer
        batch_first=False,       # input & output will has not batch size as 1st dimension. e.g. (time_step, time_step, input_size)
        bidirectional=True,
        dropout=0.5
      )
      # self.fc1 = nn.Linear(steps_n*hiddenDim*2, 10)
      # self.fc2 = nn.Linear(10, num_classes)
      # self.relu = nn.ReLU(inplace=False)
      # self.fc3 = nn.Linear(steps_n*hiddenDim*2, num_classes)

      self.fc4 = nn.Linear(steps_n*input_dim, 50)
      self.dropout = nn.Dropout(p=0.5, inplace=False)
      self.relu = nn.ReLU(inplace=False)
      self.fc5 = nn.Linear(50, num_classes)



  def forward(self, x, constant):
    debug = False

    x = GradReverse.grad_reverse(x.float(), constant)

    # Input: (seq_len, batch, input_size)
    # Output: (seq_len, batch, num_directions * hidden_size)
    x = x.transpose(0,1)
    x = x.reshape(x.size()[0], -1)
    out1 = self.relu(self.fc4(x))
    out2 = self.fc5(out1)

    if debug:
      print('DomainClassifier_lstm')
      print('out1_seq size:', out1_seq.size())
      print('out2 size:', out2.size())
      print('out3 size:', out3.size())
      
    return out2



# """
# SVM for Domain discrimination
# """
# class SVM(object):
#     def __init__(self, batch_size, vocab_size, max_seq_len=50, hidden_units=128, learning_rate=0.006):
#         self.vocab_size = vocab_size
#         self.model = None


#     def vectorize(self, sequence):
#         """ Convert sequence of word-indices into one-hot vector
#         """
#         out = [0] * self.vocab_size
#         for x in sequence:
#             if x == 0:
#                 break
#             try:
#                 out[x] += 1
#             except:
#                 pass
#         return out


#     def prepare_examples(self, x, xl, y, yl):
#         """ Convert examples into a sparse matrix of one-hot vectors
#             each vector is the concatination of x (sequence from corpus 1) and 
#             y (sequence from corpus 2)
#         """
#         return csr_matrix([self.vectorize(xi[:xli] + yi[:yli]) \
#                                for xi, xli, yi, yli in zip(x, xl, y, yl)])


#     def prepare_labels(self, d):
#         """ Translate probability distributions to binary label
#         """
#         return [np.argmax(di) for di in d]


#     def train_on_batch(self, domains, x, x_lens, y, y_lens, c=3000):
#         """ Train svm on some data
#         """
#         self.model = svm.SVC(C=c, probability=True, verbose=2)
#         examples = self.prepare_examples(x, x_lens, y, y_lens)
#         labels = self.prepare_labels(domains)
#         self.model.fit(examples, labels)


#     def predict(self, x, x_lens, y, y_lens):
#         """ Predict domains for some examples
#         """
#         examples = self.prepare_examples(x, x_lens, y, y_lens)
#         y_hat = self.model.predict(examples)
#         return y_hat


#     def mse(self, labels, x, x_lens, y, y_lens):
#         """ mean squared error (MSE)
#         """
#         examples = self.prepare_examples(x, x_lens, y, y_lens)
#         y_hat = self.model.predict_proba(examples)
#         mse = mean_squared_error(labels, y_hat)
#         return mse


#     def mae(self, labels, x, x_lens, y, y_lens):
#         """ mean absolute error (MAE)
#         """
#         examples = self.prepare_examples(x, x_lens, y, y_lens)
#         y_hat = self.model.predict_proba(examples)
#         mae = mean_absolute_error(labels, y_hat)
#         return mae


#     def fit(self, dataset):
#         """ Fit the model to a dataset and evaluate with MAE
#         """
#         dataset.set_batch_size(dataset.get_n('train') - 1)
#         train_data = next(dataset.mixed_batch_iter())

#         dataset.set_batch_size(dataset.get_n('val') - 1)
#         val_data = next(dataset.mixed_batch_iter(data='val'))
#         print 'INFO: training on ', dataset.get_n('train') - 1, ' examples'
#         self.train_on_batch(*train_data)  
#         mae = self.mae(*val_data)
#         print 'INFO: mae: ', mae


#     def test(self, dataset, mae=False):
#         """ Test the dataset on a dataset
#         """
#         dataset.set_batch_size(dataset.get_n('test') - 1)
#         test_data = next(dataset.mixed_batch_iter(data='test'))
#         if not mae:
#             return self.mse(*test_data)
#         else:
#             return self.mae(*test_data)