import agents
import engine
import logging
import numpy as np
import utility as util
import matplotlib.pyplot as plt
from scipy.special import softmax
from plot import *

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
        # Need space for each action for each agent, plus a reward vector for each agent and an action taken for each agent.
        self.history = np.zeros((N, iters, (2*self.num_actions*self.num_actors+self.num_actions)))

    def updateHistory(self, n, iter, results, use_softmax=False):
        if(use_softmax):
            for i in range(self.num_actors):
                self.history[n, iter , (i*self.num_actions):((i+1)*self.num_actions)] = \
                    softmax(self.actors[i].policy)
        else:
            for i in range(self.num_actors):
                self.history[n, iter , (i*self.num_actions):((i+1)*self.num_actions)] = \
                    self.actors[i].policy/self.actors[i].policy.sum()

        self.history[n, iter, -len(results):] = results
        
    
    def update(self):
        return self.gengine.update()
    
    def reset(self):
        for i in range(self.num_actors):
            self.actors[i].policy = self.initial_policy[i]
    
    def display(self, args, nash=None):
        f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
        plotPolicyOverTime(ax1, self.history[:,:,0:(self.num_actors*self.num_actions)], self.plot_legend, args)    
        plotTimeAveragedPolicy(ax2, self.history[:,:,0:(self.num_actors*self.num_actions)], self.plot_legend, args)
        if nash:
            plt.hlines(nash, 0-(args.n_iterations/100), args.n_iterations+(args.n_iterations/100))
        plt.show()

