class NeuralNet():

    def __init__(self,):
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