
class PolcyIncrementOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, policy, last_action, reward, state):
        policy[last_action] = policy[last_action] + reward * self.learning_rate
        return policy