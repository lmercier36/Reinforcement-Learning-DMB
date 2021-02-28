import sys
sys.path.append('..')
from utils import *

import argparse

#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class OthelloNNet(nn.Module):
    def __init__(self, game, args):
        # paramètres propres au jeu
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize() 
        self.args = args

        super(OthelloNNet, self).__init__()
        #Bloc 1 : bloc de convolution
        self.cn1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(args.num_channels)

        self.cn2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(args.num_channels)

        self.cn3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.bn3 = nn.BatchNorm2d(args.num_channels)

        self.cn4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        #Bloc 2 : FeedForward neural network
        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024) #de dimension 1024
        self.bn5 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)  #de dimension 512
        self.bn6 = nn.BatchNorm1d(512)

        # layer pour le vecteur policy
        self.fc3 = nn.Linear(512, self.action_size)

        #layer pour la output value
        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        # blec de convolution                                                            
        s = s.view(-1, 1, self.board_x, self.board_y)            
        s = F.relu(self.bn1(self.cn1(s)))
        s = F.relu(self.bn2(self.cn2(s)))
        s = F.relu(self.bn3(self.cn3(s)))
        s = F.relu(self.bn4(self.cn4(s)))
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        #bloc linéaire
        s = F.relu(self.bn5(self.fc1(s)))
        s = F.dropout(s, p=self.args.dropout, training=self.training)
        s = F.relu(self.bn6(self.fc2(s)))
        s = F.dropout(s, p=self.args.dropout, training=self.training)

        #prédiction des outputs : vecteur policy et output value
        pi = self.fc3(s)
        v = self.fc4(s)

        return F.log_softmax(pi, dim=1), torch.tanh(v)