import numpy as np
from tictactoe_gameutill import board
from tictactoe_gameutill import mcts
from tictactoe_gameutill import network
from tictactoe_gameutill import play
from tictactoe_gameutill import train

train_tictactoe = train.Trainner()

#filename = '500' + "_th_dict"
#train_tictactoe.load_model(filename)
train_tictactoe.train(1000)
#train_tictactoe.ai_play(filename)
#