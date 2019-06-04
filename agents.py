import numpy as np
from scipy.special import softmax

# Rock-paper-scissors game and agent implementation
class VectorPolicyAgent(object):
    def __init__(self, optimizer, off_policy, policy, use_softmax):
        self.policy = policy
        self.off_policy = off_policy
        self.optimizer = optimizer
        self.lastAction = None
        self.use_softmax = use_softmax

        # generate the action vector.
        self.action_vector = np.arange(len(self.policy))

        # if we are off policy generate a random policy.
        if off_policy:
            self.random_policy = np.ones(len(self.policy))/len(self.policy)
    
    def act(self, state):
        # pick an action according to the reward function defined by our vector.
        if self.off_policy:
            self.lastAction = np.random.choice(self.action_vector, p=self.random_policy)
        elif self.use_softmax: 
            self.lastAction = np.random.choice(self.action_vector, p=softmax(self.policy))
        else:
            self.lastAction = np.random.choice(self.action_vector, p=self.policy/self.policy.sum())

        return self.lastAction

    def update(self, reward, state):
        # update the policy using the optmizer iff we actual took an action last update.
        if self.lastAction is None:
            return

        self.policy = self.optimizer.optimize(self.policy, self.lastAction, reward, state)
