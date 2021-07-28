import torch.nn as nn
import torch
import torch.nn.functional as F
import NeuralNet
import os
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
        x = self.cnn(x)
        res = self.shortcut(x)
        return x + res


class Net(nn.Module):
    def __init__(self,boardsize):
        super(Net, self).__init__()

        self.boardsize = boardsize;
        self.block1 = nn.Sequential(
            conv2d(4,32,3,1,1),
            conv2d(32, 32, 3, 1,1),
            conv2d(32, 32, 3, 1, 1)
        )
        self.valueblock_1 = nn.Sequential(
            conv2d(32, 4, 1, 1, 0)
        )
        self.valueblock_2 = nn.Sequential(
            nn.Linear(4*self.boardsize*self.boardsize, 1),
            nn.Tanh()
        )
        self.policyblock_1 = nn.Sequential(
            conv2d(32, 4, 1, 1, 0)
        )
        self.policyblock_2 = nn.Sequential(
            nn.Linear(4 * self.boardsize * self.boardsize, self.boardsize * self.boardsize),
            nn.Softmax(dim = 1)
        )
    def forward(self, x):
        x = self.block1(x)
        a_probs = self.policyblock_1(x)
        a_probs = a_probs.view(-1,4 * self.boardsize * self.boardsize)
        a_probs = self.policyblock_2(a_probs)

        Q = self.valueblock_1(x)
        Q = Q.view(-1,4 * self.boardsize * self.boardsize)
        Q = self.valueblock_2(Q)
        return a_probs, Q


class Network(NeuralNet.NeuralNet) :
    def __init__(self, boardsize,model_file_path = None):
        self.boardsize = boardsize
        self.l2_const = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net(boardsize).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(),weight_decay=self.l2_const)

        if model_file_path :
            self.net.load_state_dict(torch.load(model_file_path))


    def train(self, state, mcts_probs, winner):
        state_batch = state.to(self.device).detach()
        mcts_probs_batch = mcts_probs.to(self.device).detach()
        winner_batch = winner.to(self.device).detach()
        self.optimizer.zero_grad()
        act_prob, Q = self.net(state_batch)
        log_act_prob = torch.log(act_prob)
        value_loss = F.mse_loss(Q.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs_batch * log_act_prob, 1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        return loss.detach()



    def predict(self, state):
        state = state.to(self.device)
        pi, Q = self.net(state)
        return pi.data.numpy()[0], Q.data.numpy()[0]

    def save_checkpoint(self, filename,folder = "D:/alphazero_modeldict/ticktactoe"):
        torch.save(self.net.state_dict(), f"{folder}/{filename}.pt")

    def load_checkpoint(self, filename,folder = "D:/alphazero_modeldict/ticktactoe"):
        self.net.load_state_dict(torch.load(f"{folder}/{filename}.pt"))