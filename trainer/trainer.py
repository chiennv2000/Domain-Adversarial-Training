import logging, time, sys, os
sys.path.append("..")

import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

from model import Classifier, Discriminator, FeatureExtractor
from utils.function import ReversalGradient

class Model(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, n_classes):
        super(Model, self).__init__()
        self.out_channels = out_channels
        
        self.feature_extractor = FeatureExtractor(in_channels, out_channels, kernel_size)
        self.domain_classifier = Discriminator(out_channels * 4 * 4)
        self.label_classifier = Classifier(out_channels * 4 * 4, n_classes)
    
    def forward(self, x, alpha):
        x = x.expand(x.data.shape[0], 3, 28, 28)
        feature = self.feature_extractor(x)
        
        feature = feature.view(-1, self.out_channels * 4 * 4)
        reverse_feature = ReversalGradient.apply(feature, alpha)
        
        y_pred = self.label_classifier(feature)
        domain_pred = self.domain_classifier(reverse_feature)
        
        return y_pred, domain_pred
    

class Trainer(object):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 n_classes,
                 device,
                 dataloader_source,
                 dataloader_target):
        self.model = Model(in_channels, out_channels, kernel_size, n_classes)
        
        self._optimizer = None
        self._loss_class = nn.NLLLoss()
        self._loss_domain = nn.NLLLoss()
        self._device = device
        
        self.dataloader_source = dataloader_source
        self.dataloader_target = dataloader_target
        self.num_samples = min(len(dataloader_source), len(dataloader_target))
    
    def train_step(self,
                   source_data,
                   source_labels,
                   target_data,
                   alpha):
        self.model.train()
        self._optimizer.zero_grad()
        
        src_y_pred, src_domain_pred = self.model.forward(source_data, alpha)
        _, tgt_domain_pred = self.model.forward(target_data, alpha)
        
        err_src_label = self._loss_class(src_y_pred, source_labels)
        err_src_domain = self._loss_domain(src_domain_pred, torch.zeros(src_y_pred.size(0)).long().to(self._device))
        err_tgt_domain = self._loss_domain(tgt_domain_pred, torch.ones(src_y_pred.size(0)).long().to(self._device))
        
        err = err_src_label + err_src_domain + err_tgt_domain
        
        err.backward()
        self._optimizer.step()
        
        return err
        
    def train(self, n_epochs, lr):
        self._optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(n_epochs):
            src_data_iter = iter(self.dataloader_source)
            tgt_data_iter = iter(self.dataloader_target)
            
            for i in range(self.num_samples):
                p = float(i + epoch * self.num_samples) / n_epochs / self.num_samples
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                
                src_data, src_labels = (t.to(self._device) for t in next(src_data_iter))
                tgt_data, _ = (t.to(self._device) for t in next(tgt_data_iter))
                
                loss = self.train_step(src_data, src_labels, tgt_data, alpha)
        
                print("Epoch: {}/{} - Iter {}/{} - Loss: {}".format(epoch, n_epochs, i, self.num_samples, loss))