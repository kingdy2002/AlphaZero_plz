import math
import numpy as np
import copy
import torch


class MCTsNode(object) :
    def __init__(self,parent,prior_p, c_ratio) :
        self.parent = parent
        self.children = {}
        self.visited_n = 0
        self.W = 0
        self.Q = 0
        self.U = 0
        self.P = prior_p
        self.c_ratio = c_ratio
        self.update_U()

    def update_U(self):
        if self.parent is not None :
            self.U = self.Q + self.c_ratio*self.P*math.sqrt(self.parent.visited_n)/(1+self.visited_n)

    def update(self,child_q) :
        self.visited_n += 1
        self.W += child_q
        self.Q = self.W/self.visited_n
        self.update_U()

    def update_all(self,child_q) :
        if self.parent :
            self.parent.update(-child_q)
        self.update(child_q)

    def search(self) :
        return max(self.children.items(), key = lambda child : child[1].U)

    def add_chiled(self, action_prob, valid_move) :
        for move in valid_move:
            if move not in self.children :
                self.children[move] = MCTsNode(self,action_prob[move],self.c_ratio)

    def is_leaf(self):
        return self.children == {}

    def is_root(self):
        return self.parent is None

class MCTs(object) :
    def __init__(self, network, c_ratio, n_playout) :
        self.root = MCTsNode(None,1,c_ratio)
        self.network = network
        self.n_playout = n_playout
        self.c_ratio = c_ratio


    def playout(self,board) :
        self.node = self.root
        while not self.node.is_leaf() :
            move, self.node = self.node.search
            board.place(move)
            end, win = board.check_end()
            if end :
                child_q = win
            else :
                with torch.no_grad():
                    a_probs, child_q = self.network(board.current_state())
                self.node.add_chiled(a_probs)
            self.node.update_all(child_q)

    def get_move_probs(self,board):
        for i in range(self.n_playout):
            board_copy = copy.deepcopy(board)
            self.playout(board_copy)
        act_visits = [(act, node.visited_n)
                      for act, node in self.root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = visits/np.sum(visits)

        return acts, act_probs

    def update_and_restart_mcts_by_move(self,move):
        self.root = self.root.children[move]
        self.root.parent = None

    def reset_mcts(self):
        self.root = MCTsNode(None, 1, self.c_ratio)

    def get_move(self,board,self_play):
        acts, probs = self.get_move_probs(board)
        move_probs = np.zeros(board.width * board.height)
        move_probs[list(acts)] = probs
        if self_play :
            move = np.random.choice(
                acts,
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.03 * np.ones(len(probs)))
            )
            self.update_and_restart_mcts_by_move(move)
        else :
            move = np.random.choice(acts, p=probs)
            self.reset_mcts()

        return move, move_probs