from game import Game
import engine
import numpy as np

ROCK = 0
PAPER = 1
SCISSORS = 2

#           Rock        Paper       Scissors
#Rock       0           -1          1
#Paper      1           0           -1
#Scissors   -1          1           0

PAYOUT_MATRIX = np.array([[0, -1,  1],
                          [1,  0, -1],
                          [-1, 1,  0]])

POLICY_LEGEND =  ['agent1 rock', 'agent1 paper', 'agent1 scissors', 'agent2 rock', 'agent2 paper', 'agent2 scissors']

class RPS(Game):
    def __init__(self, args, num_actors=2, num_actions=3):
        Game.__init__(self, args, num_actors, num_actions)
        self.payout_matrix = PAYOUT_MATRIX
        self.plot_legend = POLICY_LEGEND
        self.gengine = engine.EpisodicGame(self.actors, self.payout_matrix)

    def display(self, args):
        super().display(args, .33)
