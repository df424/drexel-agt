
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

# Rock-paper-scissors game and agent implementation
class RPSAgent(object):
    def __init__(self, optimizer, policy=None):
        # if we weren't given an initial policy generate one randomly
        if policy is None:
            self.policy = np.random.rand(3)
        else:
            self.policy = policy

        self.optimizer = optimizer
        self.lastAction = None
    
    def getNormalizedPolicy(self):
        """Normalize the policy into a probability"""
        p = np.copy(self.policy)
        if(p.min() < 0):
            p = p + (-1*p.min())
        return p/p.sum()

    def act(self, state):
        # pick an action according to the reward function defined by our vector.
        self.lastAction = np.random.choice([0, 1, 2], p=self.getNormalizedPolicy())
        return self.lastAction

    def update(self, reward, state):
        # update the policy using the optmizer iff we actual took an action last update.
        if not self.lastAction:
            return

        self.policy = self.optimizer.optimize(self.policy, self.lastAction, reward, state)

class RPSGame(object):
    def __init__(self, agents):
        self.agents = agents

    def update(self):
        # assume two agents.
        a1 = self.agents[0].act(None)
        a2 = self.agents[1].act(None)

        r1 = PAYOUT_MATRIX[a1, a2]
        r2 = PAYOUT_MATRIX[a2, a1]
        self.agents[0].update(r1, None)
        self.agents[1].update(r2, None)

        return ([a1, a2], [r1, r2])