import torch.nn as nn
import numpy as np
import torch
import torch.functional as F
import NeuralNet

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class conv2d(nn.Module) :
    def __init__(self,in_ch,out_ch,kernel_size,stride,padding):
        super(conv2d,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True)
        )
    def forward(self,x) :
        x = self.block(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.cnn = conv2d(in_ch,out_ch,kernel_size,stride,padding)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        res = x
        x = self.cnn(x)
        x += self.shortcut(res)
        x = nn.ReLU(True)(x)
        return x


class Net(nn.Module):
    def __init__(self,boardsize):
        super(Net, self).__init__()

        self.boardsize = boardsize;
        self.block1 = nn.Sequential(
            conv2d(3,64,3,1,1),
            conv2d(64, 128, 3, 1,1)
        )
        self.block2 = nn.Sequential(
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128)
        )
        self.valueblock = nn.Sequential(
            conv2d(128, 4, 1, 1, 0),
            nn.Linear(4*self.boardsize*self.boardsize, self.boardsize*self.boardsize),
            nn.ReLU(),
            nn.Linear(self.boardsize * self.boardsize, 1),
            nn.Tanh()
        )
        self.policyblock = nn.Sequential(
            conv2d(128, 4, 1, 1, 0),
            nn.Linear(4 * self.boardsize * self.boardsize, self.boardsize * self.boardsize),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Network(NeuralNet.NeuralNet) :
    def __init__(self, game):
        pass

    def train(self, state, mcts_probs, winner):
        """
        get self play data (state, mcts_probs, winner) and update
        itself Network
        """
        pass

    def predict(self, state):
        """
        Input:
            board.currentstate data format

        Returns:
            pi: a policy vector for the current board- a numpy array of length
                game.getActionSize
            v: a float in [-1,1] that gives the value of the current board
        """
        pass

    def save_checkpoint(self, folder, filename):
        """
        Saves the current neural network (with its parameters) in
        folder/filename
        """
        pass

    def load_checkpoint(self, folder, filename):
        """
        Loads parameters of the neural network from folder/filename
        """
        pass