import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet
from torch.nn import Softmax

class Net(nn.Module):
    def __init__(self, num_classes, feature_extract=False, use_pretrained=True):
        super().__init__()
        self.net = alexnet(pretrained=use_pretrained)
        self.set_parameter_requires_grad(feature_extract)
        #self.net.classifier[1] = nn.Linear(9216,2048)
        #self.net.classifier[4] = nn.Linear(2048,1024)
        #num_ftrs = self.net.classifier[6].in_features
        self.net.classifier[6] = nn.Linear(self.net.classifier[6].in_features, num_classes)

    def set_parameter_requires_grad(self, feature_extracting):
        if feature_extracting:
            for param in self.net.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.net(x)
        return x

    def summary(self):
        print(self.net)
