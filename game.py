import agents
import engine
import logging
import numpy as np
import utility as util

class Game:
    def __init__(self, args, num_actors, actions):
        self.num_actors = num_actors
        self.actions = actions
        self.payout_matrix = np.empty(shape=(10,10))
        self.initial_policy = []
        for i in range(num_actors):
            self.initial_policy.append(util.initiallzePolicy(actions, args))
        self.actors = []
        optimizer = util.initializeOptimizer(args)
        for i in range(num_actors):
            self.actors.append(agents.VectorPolicyAgent(
                    optimizer, args.off_policy, self.initial_policy[i], args.use_softmax)
                )
        
        self.gengine = None
    
    def update(self):
        self.gengine.update()
    
    def reset(self):
        for i in range(self.num_actors):
            self.actors[i].policy = self.initial_policy[i]
    
    # def display(self, args):
    #     if(args.use_softmax):

    #         history[n, i,0:3] = softmax(agent1.policy)
    #         history[n, i,3:6] = softmax(agent2.policy)
    #     else:
    #         history[n, i,0:3] = agent1.policy/agent1.policy.sum()
    #         history[n, i,3:6] = agent2.policy/agent2.policy.sum()

