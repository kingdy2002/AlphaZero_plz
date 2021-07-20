import numpy as np
from gomouk_gameutill import board
from gomouk_gameutill import mcts
from gomouk_gameutill import network
from gomouk_gameutill import play
from gomouk_gameutill import train

train_gomouk = train.Trainner()
train_gomouk.train(1000)
