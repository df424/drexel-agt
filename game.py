import agents
import engine
import logging
import numpy as np
import utility as util
import matplotlib.pyplot as plt
from scipy.special import softmax
from plot import *

class GameHistory:
    def __init__(self, num_samples, num_iterations, num_actors, num_actions):
        self._historys = []

        for _ in range(num_actors):
            self._historys.append({
                'policy':np.zeros((num_samples, num_iterations, num_actions)),
                'rewards':np.zeros((num_samples, num_iterations, num_actions)),
                'actions':np.zeros((num_samples, num_iterations))
            })
    
    def __getitem__(self, key):
        return self._historys[key]

    def __len__(self):
        return len(self._historys)
            

class Game:
    def __init__(self, args, num_actors, num_actions):
        self.num_actors = num_actors
        self.num_actions = num_actions
        self.initial_policy = []
        for i in range(num_actors):
            self.initial_policy.append(util.initiallzePolicy(num_actions, args))
        self.actors = []
        optimizer = util.initializeOptimizer(args)
        for i in range(num_actors):
            self.actors.append(
                agents.VectorPolicyAgent(
                    optimizer, args.off_policy, self.initial_policy[i], args.use_softmax
                    )
                )
        
        self.payout_matrix = None
        self.plot_legend = None
        self.gengine = None
        self.history = None

    def initHistory(self, N, iters):
        # Allocate the history object.
        self.history = GameHistory(N, iters, self.num_actors, self.num_actions)

    def updateHistory(self, n, iter, actions, rewards, use_softmax=False):
        for i in range(self.num_actors):
            if(use_softmax):
                self.history[i]['policy'][n, iter] = softmax(self.actors[i].policy)
            else:
                self.history[i]['policy'][n, iter] = self.actors[i].policy/self.actors[i].policy.sum()

            self.history[i]['actions'][n, iter] = actions[i]
            self.history[i]['rewards'][n, iter] = rewards[i]
    
    def update(self):
        return self.gengine.update()
    
    def reset(self):
        for i in range(self.num_actors):
            self.actors[i].policy = self.initial_policy[i]
    
    def display(self, args, nash=None):
        f,(ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        plotPolicyOverTime(ax1, self.history, self.plot_legend, args)    
        plotTimeAveragedPolicy(ax2, self.history, self.plot_legend, args)
        #if nash:
            #plt.hlines(nash, 0-(args.n_iterations/100), args.n_iterations+(args.n_iterations/100))
        plotRegret(ax3, self.history, ['A1', 'A2'], args)
        plt.show()

