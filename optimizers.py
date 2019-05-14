
class PolcyIncrementOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, policy, last_action, reward, state):
        return policy + self.learning_rate * (reward - policy)

class MultiplicitiveWeightOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, policy, last_action, reward, state):
        return policy * (1+self.learning_rate*reward)