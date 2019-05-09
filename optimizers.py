
class PolcyIncrementOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def optimize(self, policy, last_action, reward, state):
        policy[last_action] = policy[last_action] + self.learning_rate * (reward - policy[last_action])