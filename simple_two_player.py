import numpy as np
from scipy.special import softmax

# Rock-paper-scissors game and agent implementation
class VectorPolicyAgent(object):
    def __init__(self, optimizer, off_policy, policy):
        self.policy = policy
        self.off_policy = off_policy
        self.optimizer = optimizer
        self.lastAction = None

        # generate the action vector.
        self.action_vector = np.arange(len(self.policy))

        # if we are off policy generate a random policy.
        if off_policy:
            self.random_policy = np.ones(len(self.policy))/len(self.policy)
    
    def act(self, state):
        # pick an action according to the reward function defined by our vector.
        if self.off_policy:
            self.lastAction = np.random.choice(self.action_vector, p=self.random_policy)
        else: # must be on policy...
            #self.lastAction = np.random.choice(self.action_vector, p=softmax(self.policy))
            self.lastAction = np.random.choice(self.action_vector, p=self.policy/self.policy.sum())

        return self.lastAction

    def update(self, reward, state):
        # update the policy using the optmizer iff we actual took an action last update.
        if self.lastAction is None:
            return

        self.policy = self.optimizer.optimize(self.policy, self.lastAction, reward, state)

class EpisodicGame(object):
    def __init__(self, agents, payout_matrix):
        self.agents = agents
        self.payout_matrix = payout_matrix

    def update(self):
        # assume two agents.
        a1 = self.agents[0].act(None)
        a2 = self.agents[1].act(None)

        # generate a vector of rewards.
        r1 = np.zeros(len(self.agents[0].policy))
        r2 = np.zeros(len(self.agents[1].policy))

        for i in range(len(r1)):
            r1[i] = self.payout_matrix[i, a2]
            r2[i] = self.payout_matrix[i, a1]

        self.agents[0].update(r1, None)
        self.agents[1].update(r2, None)

        return ([a1, a2], [r1, r2])