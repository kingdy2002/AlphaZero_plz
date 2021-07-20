import numpy as np
import random

class randomplayer(object):
    def __init__(self,board):
        self.board = board

    def random_action(self):
        self.valid = self.board.valid_move
        move = random.choice(list(self.valid))
        return move