
class PolcyIncrementOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, policy, last_action, reward, state):
        return policy[last_action] + self.learning_rate * (reward[last_action] - policy[last_action])

class MultiplicitiveWeightOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, policy, last_action, reward, state):
        return policy * (1+self.learning_rate*reward)