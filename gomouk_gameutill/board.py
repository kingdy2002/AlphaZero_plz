import numpy as np
import Game
import time
#player는 -1, 1
class board(Game.Board) :
    def __init__(self, width, height) :
        self.width = width
        self.height = height
        self.stone_n = 0
        self.state = {}
        self.players = [1,-1]
        self.valid_move = set(range(0,width*height))

    def init_board(self):
        self.state = {}
        self.stone_n = 0
        self.last_move = -1
        self.current_player = 1
        self.valid_move = set(range(0, self.width * self.height))

    def place(self,move):
        self.state[move] = self.current_player
        self.stone_n += 1
        self.last_move = move
        self.valid_move.remove(move)
        self.current_player = -self.current_player

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def locaton_to_move(self, location):
        h = location[0]
        w = location[1]
        return h* self.width + w

    def valid_move(self):
        valid = np.zeros(self.width * self.height)
        for i in self.valid_move:
            valid[i] = 1
        return valid


    def has_win(self,length):

        width = self.width
        height = self.height
        states = self.state
        n = length

        moved = list(set(range(width * height)) - self.valid_move)
        if len(moved) < n * 2-1:
            return False, 0
        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, 0) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, 0) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, 0) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, 0) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, 0

    def check_end(self):
        win, winner = self.has_win(5)
        if winner :
            if winner == self.current_player :
                return (True, 1)

            else :
                return (True, -1)

        if self.stone_n > self.width * self.height - 5 :
            return (True , 0)
        else :
            return (False, 0)

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.state:
            moves, players = np.array(list(zip(*self.state.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        if len(self.state) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state


class Game(Game.game):

    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.state.get(loc, 0)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_self_play(self,MCTs,visiable):
        self.board.init_board()
        MCTs.reset_mcts(self.board)
        states, mcts_probs,current_players = [], [], []
        while True :
            move, move_probs = MCTs.get_move(self.board,True)
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            self.board.place(move)
            states.append(self.board.current_state())
            MCTs.update_and_restart_mcts_by_move(move,self.board)
            end, win = self.board.check_end()

            if visiable :
                self.graphic(self.board, 1, -1)

            if end :
                if win == 1 :
                    winner = self.board.current_player
                elif win == -1 :
                    winner = -self.board.current_player
                else :
                    winner = 0
                winners_z = np.zeros(len(current_players))
                if winner != 0:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                MCTs.reset_mcts(self.board)
                return winner, zip(states, mcts_probs, winners_z)

    def ai_play(self,MCTs):
        self.board.init_board()
        MCTs.reset_mcts(self.board)
        while True :
            move, move_probs = MCTs.get_move(self.board,False)
            self.board.place(move)
            MCTs.update_and_restart_mcts_by_move(move,self.board)
            end, win = self.board.check_end()
            self.graphic(self.board, 1, -1)
            if end :
                return