class Board():

    def __init__(self):
        pass

    def init_borad(self):
        """
        return :
        init borad

        """
        pass

    def place(self,move):
        """
        input:
            move that is action (encode by 0 ~ weigh*height)
            change state changing all of things follow state

        """
        pass

    def valid_move(self):

        """
        return:
            valid position composed by np.array
        """
        pass

    def has_win(self, args):
        """
        input:
            args is condition of the game

        return:
            if game finish return player who win the game
            else return 0
        """
        pass

    def check_end(self):
        """
        return:
            tuple that composed by indicate game was end and who is win
            (True or False(end is True), 1(if player win) 0(nobody win) -1(other win))
        """

        pass

    def current_state(self):

        """
        return:
            change state to input data form of Nueral net and return
        """
        pass

class game(object) :
    def __init__(self):
        pass
    def graphic(self):
        """
        display board
        """
        pass

    def start_self_play(self,MCTs):
        """
        self play and collect self play data

        return :
            winner, each step's states, action_probs
        """