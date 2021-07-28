import numpy as np
from gomouk_gameutill import board
from gomouk_gameutill import mcts
from gomouk_gameutill import network
from gomouk_gameutill import play
from collections import defaultdict, deque
import random
import torch

class Trainner(object):
    def __init__(self) :
        self.board_width = 8
        self.board_height = 8
        self.board = board.board(self.board_width,self.board_height)
        self.game = board.Game(self.board)
        self.network = network.Network(self.board_width)
        self.mct = mcts.MCTs(self.network.net, 5, 800)
        self.buffer_size = 1000
        self.batch_size = 64
        self.data_buffer = deque(maxlen=self.buffer_size)

    def collect_data(self):
        winner, play_data = self.game.start_self_play(self.mct,False)
        self.data_buffer.extend(play_data)

    def update(self):
        loss = 0
        update = False
        if len(self.data_buffer) > 500 :
            mini_batch = random.sample(self.data_buffer, self.batch_size)
            state_batch = [data[0] for data in mini_batch]
            mcts_probs_batch = [data[1] for data in mini_batch]
            winner_batch = [data[2] for data in mini_batch]

            state_batch = torch.FloatTensor(state_batch).cuda()
            mcts_probs_batch = torch.FloatTensor(mcts_probs_batch).cuda()
            winner_batch = torch.FloatTensor(winner_batch).cuda()

            loss = self.network.train(state_batch,mcts_probs_batch,winner_batch)
            update = True
        return loss, update


    def train(self,episode_n):
        best_win_ratio = 0
        win_ratio_list = []
        for i in range(episode_n) :
            self.collect_data()
            i = i+ 500
            print("episode ", i," finsih")
            loss, update = self.update()
            if i % 10 == 0 :
                filename = str(i)+"_th_dict"
                self.network.save_checkpoint(filename)
                #win_ratio = self.evaluate()
                #win_ratio_list.append(win_ratio)
                #print("episode ",i," : win ratio is ",win_ratio)
                #if best_win_ratio < win_ratio :
                #    self.network.save_checkpoint("best_dict")

    def load_model(self,filename):
        self.network.load_checkpoint(filename)

    def ai_play(self, filename):
        self.network.load_checkpoint(filename)
        winner, play_data = self.game.start_self_play(self.mct, True)