from game import Game
import engine
import numpy as np

TURN = 0
STAY = 1
#          them
# me    TURN   STAY
#TURN     0     -2
#STAY     2     -10

PAYOUT_MATRIX = np.array([[0, -.2],
                          [.2,  -1.0]])

POLICY_LEGEND = ['agent1 turn', 'agent1 stay', 'agent2 turn', 'agent2 stay']

class Chicken(Game):
    def __init__(self, args, num_actors=2, num_actions=2):
        Game.__init__(self, args, num_actors, num_actions)
        self.payout_matrix = PAYOUT_MATRIX
        self.plot_legend = POLICY_LEGEND
        self.gengine = engine.EpisodicGame(self.actors, self.payout_matrix)
