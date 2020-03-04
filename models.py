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


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
  def __init__(self, num_classes=10, input_dim=50):
      super(ConvNet, self).__init__()
      self.layer1 = nn.Sequential(
          nn.Conv1d(3, 16, kernel_size=3, stride=1, padding=2),
          nn.BatchNorm1d(16),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      self.layer2 = nn.Sequential(
          nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=2),
          nn.BatchNorm1d(32),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      self.layer3 = nn.Sequential(
          nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      
      cnn_layer1_dim = (input_dim+2*2-1*(3-1)-1)+1
      pool_layer1_dim = (cnn_layer1_dim-1*(2-1)-1)/2+1

      cnn_layer2_dim = (pool_layer1_dim+2*2-1*(3-1)-1)+1
      pool_layer2_dim = (cnn_layer2_dim-1*(2-1)-1)/2+1

      cnn_layer3_dim = (pool_layer2_dim+2*2-1*(3-1)-1)+1
      pool_layer3_dim = (cnn_layer3_dim-1*(2-1)-1)/2+1

      # print('cnn_layer1_dim:', cnn_layer1_dim)
      # print('pool_layer1_dim:', pool_layer1_dim)
      # print('cnn_layer2_dim:', cnn_layer2_dim)
      # print('pool_layer2_dim:', pool_layer2_dim)
      # print('cnn_layer3_dim:', cnn_layer3_dim)
      # print('pool_layer3_dim:', pool_layer3_dim)
      # fc_dim = int(((((input_dim)+2*2-1)/2+2*2-1)/2+2*2-1)/2*64)
      self.fc = nn.Linear(int(pool_layer3_dim)*64, num_classes)
      
  def forward(self, x):
    out1 = self.layer1(x.float())
    # print('out1 size:', out1.size())
    out2 = self.layer2(out1)
    # print('out2 size:', out2.size())
    out3 = self.layer3(out2)
    # print('out3 size:', out3.size())
    out3 = out3.reshape(out3.size(0), -1)
    # print('out3 size:', out3.size())
    out4 = self.fc(out3)
    # print('x, out1, out2, out 3, out4 size',  x.size(), out1.size(), out2.size(), out3.size(), out4.size())
    return out4


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
          nn.Conv1d(3, channel_n, kernel_size=3, stride=1, padding=2),
          nn.BatchNorm1d(channel_n),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      self.layer2 = nn.Sequential(
          nn.Conv1d(channel_n, channel_n*2, kernel_size=3, stride=1, padding=2),
          nn.BatchNorm1d(channel_n*2),
          nn.ReLU(),
          nn.MaxPool1d(kernel_size=2, stride=2))
      # self.layer3 = nn.Sequential(
      #     nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2),
      #     nn.BatchNorm1d(64),
      #     nn.ReLU(),
      #     nn.MaxPool1d(kernel_size=2, stride=2))
      
      cnn_layer1_dim = (input_dim+2*2-1*(3-1)-1)+1
      pool_layer1_dim = (cnn_layer1_dim-1*(2-1)-1)/2+1

      cnn_layer2_dim = (pool_layer1_dim+2*2-1*(3-1)-1)+1
      pool_layer2_dim = (cnn_layer2_dim-1*(2-1)-1)/2+1

      # cnn_layer3_dim = (pool_layer2_dim+2*2-1*(3-1)-1)+1
      # pool_layer3_dim = (cnn_layer3_dim-1*(2-1)-1)/2+1

      # print('cnn_layer1_dim:', cnn_layer1_dim)
      # print('pool_layer1_dim:', pool_layer1_dim)
      # print('cnn_layer2_dim:', cnn_layer2_dim)
      # print('pool_layer2_dim:', pool_layer2_dim)
      # print('cnn_layer3_dim:', cnn_layer3_dim)
      # print('pool_layer3_dim:', pool_layer3_dim)
      # fc_dim = int(((((input_dim)+2*2-1)/2+2*2-1)/2+2*2-1)/2*64)
      # self.fc = nn.Linear(int(pool_layer2_dim)*32, num_classes)
      
  def forward(self, x):
    out1 = self.layer1(x.float())
    # print('out1 size:', out1.size())
    out2 = self.layer2(out1)
    # print('out2 size:', out2.size())
    # out3 = self.layer3(out2)
    # print('out3 size:', out3.size())
    # out3 = out3.reshape(out3.size(0), -1)
    out2 = out2.reshape(out2.size(0), -1)
    # print('out3 size:', out3.size())
    # out3 = self.fc(out2)
    # print('x, out1, out2, out 3, out4 size',  x.size(), out1.size(), out2.size(), out3.size(), out4.size())
    return out2

# fall classifier neural network (fc layers)
class ClassClassifier(nn.Module):
  def __init__(self, num_classes=10, input_dim=50):
      super(ClassClassifier, self).__init__()
      self.fc = nn.Linear(input_dim, num_classes)
      
  def forward(self, x):
    # out1 = F.relu(self.fc(x))
    out1 = self.fc(x.float())
    return out1

# domain classifier neural network (fc layers)
class DomainClassifier(nn.Module):
  def __init__(self, num_classes=10, input_dim=50):
      super(DomainClassifier, self).__init__()
      self.fc = nn.Linear(input_dim, num_classes)
      
  def forward(self, x, constant):
    out1 = GradReverse.grad_reverse(x.float(), constant)
    # out2 = F.relu(self.fc(out1))
    out2 = self.fc(out1)
    return out2

class CascadedModel(nn.Module):
  def __init__(self, modelA, modelB):
    super(CascadedModel, self).__init__()
    self.modelA = modelA
    self.modelB = modelB
      
  def forward(self, x):
    out1 = self.modelA(x.float())
    out2 = self.modelB(out1)
    return out2


class DannModel(nn.Module):
  def __init__(self, device, class_N=2, domain_N=2, channel_n=16, input_dim=10):
    super(DannModel, self).__init__()
    self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n).to(device).float()
    cnn_layer1_dim = (input_dim+2*2-1*(3-1)-1)+1
    pool_layer1_dim = (cnn_layer1_dim-1*(2-1)-1)/2+1

    cnn_layer2_dim = (pool_layer1_dim+2*2-1*(3-1)-1)+1
    pool_layer2_dim = (cnn_layer2_dim-1*(2-1)-1)/2+1

    feature_out_dim = int(pool_layer2_dim*channel_n*2)

    self.class_classfier = ClassClassifier(num_classes=class_N, input_dim=feature_out_dim).to(device).float()
    self.domain_classifier = DomainClassifier(num_classes=domain_N, input_dim=feature_out_dim).to(device).float()
      
  def forward(self, x):
    feature_out = self.feature_extractor(x)
    class_output = self.class_classfier(feature_out)
    domain_output = self.domain_classifier(feature_out, 1)
    # return feature_out
    return feature_out, class_output, domain_output

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# dann = DannModel(device, class_N=2, domain_N=2, input_dim=66).to(device).float()

class BaselineModel(nn.Module):
  def __init__(self, device, class_N=2, channel_n=16, input_dim=10):
    super(BaselineModel, self).__init__()
    self.feature_extractor = FeatureExtractor(input_dim=input_dim, channel_n=channel_n).to(device).float()
    cnn_layer1_dim = (input_dim+2*2-1*(3-1)-1)+1
    pool_layer1_dim = (cnn_layer1_dim-1*(2-1)-1)/2+1

    cnn_layer2_dim = (pool_layer1_dim+2*2-1*(3-1)-1)+1
    pool_layer2_dim = (cnn_layer2_dim-1*(2-1)-1)/2+1

    feature_out_dim = int(pool_layer2_dim*channel_n*2)

    self.class_classfier = ClassClassifier(num_classes=class_N, input_dim=feature_out_dim).to(device).float()
    # self.domain_classifier = DomainClassifier(num_classes=domain_N, input_dim=feature_out_dim).to(device).float()
      
  def forward(self, x):
    feature_out = self.feature_extractor(x)
    # print('feature_out size', feature_out.size())
    class_out = self.class_classfier(feature_out)
    # domain_output = self.domain_classifier(feature_out, 1)
    # return feature_out
    return feature_out, class_out

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model = BaselineModel(device, class_N=2, channel_n=10, input_dim=66).to(device).float()
