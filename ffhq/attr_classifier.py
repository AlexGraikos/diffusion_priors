import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Code adapted from https://github.com/Hawaii0821/FaceAttr-Analysis

class FeatureExtraction(nn.Module):
    def __init__(self, pretrained):
        super(FeatureExtraction, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)   
        self.model = nn.Sequential(*list(self.model.children())[:-1])
    def forward(self, image):
        return self.model(image)

class FeatureClassfier(nn.Module):
    def __init__(self, selected_attrs, input_dim=512, output_dim=1):
        super(FeatureClassfier, self).__init__()

        self.attrs_num = len(selected_attrs)
        self.selected_attrs = selected_attrs
        output_dim = len(selected_attrs)
        """build full connect layers for every attribute"""
        self.fc_set = {}

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        res = self.fc(x)
        return res

class FaceAttrModel(nn.Module):
    def __init__(self, pretrained, selected_attrs):
        super(FaceAttrModel, self).__init__()
        self.featureExtractor = FeatureExtraction(pretrained)
        self.featureClassfier = FeatureClassfier(selected_attrs, input_dim=512)
    
    def forward(self, image):
        features = self.featureExtractor(image)
        results = self.featureClassfier(features)
        return results
