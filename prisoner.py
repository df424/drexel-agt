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

POLICY_LEGEND = ['agent1 cooperate', 'agent1 defect', 'agent1 cooperate', 'agent2 defect']

class Prisoner(Game):
    def __init__(self, args, num_actors=2, actions=2):
        Game.__init__(self, args, num_actors, actions)
        self.payout_matrix = PAYOUT_MATRIX
        self.gengine = engine.EpisodicGame(self.actors, self.payout_matrix)
