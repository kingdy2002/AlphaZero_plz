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
        self.board_width = 10
        self.board_height = 10
        self.board = board.board(self.board_width,self.board_height)
        self.game = board.Game(self.board)
        self.network = network.Network(self.board_width)
        self.mct = mcts.MCTs(self.network.net, 2, 5)
        self.buffer_size = 10000
        self.batch_size = 64
        self.data_buffer = deque(maxlen=self.buffer_size)

    def collect_data(self):
        winner, play_data = self.game.start_self_play(self.mct,False)
        self.data_buffer.extend(play_data)

    def update(self):
        loss = 0
        update = False
        if len(self.data_buffer) > 200 :
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

    def evaluate(self):
        print("start evaluate")
        player = np.array([1, -1])
        ai = np.random.choice(player)
        self.randomplayer = play.randomplayer(self.board)
        self.board.init_borad()
        self.mct.reset_mcts(self.board)
        ai_win_count = 0

        for i in range(10) :
            self.board.init_borad()
            self.mct.reset_mcts(self.board)



            if ai == 1 :
                while True :
                    move, move_probs = self.mct.get_move(self.board,False)
                    self.board.place(move)
                    self.mct.update_and_restart_mcts_by_move(move, self.board)
                    end, win = self.board.check_end()
                    if end :
                        break
                    move = self.randomplayer.random_action()
                    self.board.place(move)
                    self.mct.update_and_restart_mcts_by_move(move, self.board)
                    end, win = self.board.check_end()
                    if end :
                        break

            if ai == -1:
                while True:
                    move = self.randomplayer.random_action()
                    self.board.place(move)
                    self.mct.update_and_restart_mcts_by_move(move, self.board)
                    end, win = self.board.check_end()
                    if end:
                        break
                    move, move_probs = self.mct.get_move(self.board, True)
                    self.board.place(move)
                    self.mct.update_and_restart_mcts_by_move(move, self.board)
                    end, win = self.board.check_end()
                    if end:
                        break
            
            if win != 0:
                if ai == self.board.current_player * win:
                    ai_win_count += 1


        return ai_win_count/33*100

    def train(self,episode_n):
        best_win_ratio = 0
        win_ratio_list = []
        for i in range(episode_n) :
            self.collect_data()
            print("episode ", i," finsih")
            loss, update = self.update()
            if episode_n % 10 == 0 :
                filename = str(i)+"_th_dict"
                self.network.save_checkpoint(filename)
                win_ratio = self.evaluate()
                win_ratio_list.append(win_ratio)
                print("episode ",i," : win ratio is ",win_ratio)
                if best_win_ratio < win_ratio :
                    self.network.save_checkpoint("best_dict")

