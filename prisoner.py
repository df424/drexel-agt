from game import Game
import engine
import numpy as np

COOPERATE = 0
DEFECT = 1

#           COOPERATE   DEFECT
#COOPERATE  -1          -10
#DEFECT      0          -4

PAYOUT_MATRIX = np.array([[-0.1, -1.0],
                          [0,  -0.4]])

POLICY_LEGEND = ['agent1 cooperate', 'agent1 defect', 'agent2 cooperate', 'agent2 defect']

class Prisoner(Game):
    def __init__(self, args, num_actors=2, num_actions=2):
        Game.__init__(self, args, num_actors, num_actions)
        self.payout_matrix = PAYOUT_MATRIX
        self.plot_legend = POLICY_LEGEND
        self.gengine = engine.EpisodicGame(self.actors, self.payout_matrix)
