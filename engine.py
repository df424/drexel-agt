import numpy as np

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

        return np.concatenate(([a1], [a2], r1, r2))