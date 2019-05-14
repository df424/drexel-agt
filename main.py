
import argparse
import rps as rps
import prisoner as pris
import numpy as np
import simple_two_player as stpg
import matplotlib.pyplot as plt
import math
from optimizers import PolcyIncrementOptimizer, MultiplicitiveWeightOptimizer
from scipy.special import softmax

def _getInitialPolicy(n_actions, args):
    if(args.random_start):
        return np.random.rand(n_actions)*abs(args.random_start_max-args.random_start_min)+args.random_start_min

    return np.zeros(n_actions) 

def _getOptimizer(args):
    if args.optimizer == 'multi-w':
        return MultiplicitiveWeightOptimizer(args.learning_rate)
    elif args.optimizer == 'td':
        return PolcyIncrementOptimizer(args.learning_rate)

    return None

def plotPolicyOverTime(axis, policy_history, legend, args):
    axis.plot(policy_history[:])   
    axis.set_title('Policy vs. Iterations, λ=' + str(args.learning_rate) + ', N=' + str(args.N))
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Normalized Policy')
    axis.legend(legend)

def plotTimeAveragedPolicy(axis, policy_history, legend, args):
    axis.plot(np.cumsum(policy_history, axis=0)/np.arange(1,len(policy_history)+1).reshape((len(policy_history),1)))
    axis.set_title('Time Averaged Policies, λ=' + str(args.learning_rate) + ', N=' + str(args.N))
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Time Averaged Policy')
    axis.grid()
    axis.legend(legend)

def rock_paper_scissors(args):
    inital_policy1 = _getInitialPolicy(3, args)
    inital_policy2 = _getInitialPolicy(3, args)
    agent1 = stpg.VectorPolicyAgent(_getOptimizer(args), args.off_policy, inital_policy1, args.use_softmax)
    agent2 = stpg.VectorPolicyAgent(_getOptimizer(args), args.off_policy, inital_policy2, args.use_softmax)
    game = stpg.EpisodicGame([agent1, agent2], rps.PAYOUT_MATRIX)

    history = np.zeros((args.n_iterations,6))
    for n in range(args.N):
        print(str(round(n*100.0/args.N)) + '%...')
        # Reset the agents initial policy...
        agent1.policy = inital_policy1
        agent2.policy = inital_policy2
        for i in range(args.n_iterations):
            #update the game
            results = game.update()

            # print if verbosity is on.
            if(args.verbose):
                print(i, agent1.policy, softmax(agent1.policy), agent2.policy, softmax(agent2.policy), results)

            if(args.use_softmax):
                history[i,0:3] = history[i,0:3] + softmax(agent1.policy)
                history[i,3:6] = history[i,3:6] + softmax(agent2.policy)
            else:
                history[i,0:3] = history[i,0:3] + agent1.policy/agent1.policy.sum()
                history[i,3:6] = history[i,3:6] + agent2.policy/agent2.policy.sum()

    f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plotPolicyOverTime(ax1, history[:,0:6]/args.N, rps.POLICY_LEGEND, args)    
    plotTimeAveragedPolicy(ax2, history[:,0:6]/args.N, rps.POLICY_LEGEND, args)
    plt.show()

def prisoners_dilema(args):
    inital_policy1 = _getInitialPolicy(2, args)
    inital_policy2 = _getInitialPolicy(2, args)
    agent1 = stpg.VectorPolicyAgent(_getOptimizer(args), args.off_policy, inital_policy1, args.use_softmax)
    agent2 = stpg.VectorPolicyAgent(_getOptimizer(args), args.off_policy, inital_policy2, args.use_softmax)
    game = stpg.EpisodicGame([agent1, agent2], pris.PAYOUT_MATRIX)

    history = np.zeros((args.n_iterations,4))

    for n in range(args.N):
        print(str(round(n*100.0/args.N)) + '%...')
        # Reset the agents initial policy...
        agent1.policy = inital_policy1
        agent2.policy = inital_policy2
        for i in range(args.n_iterations):
            #update the game
            results = game.update()

            # print if verbosity is on.
            if(args.verbose):
                print(i, agent1.policy, softmax(agent1.policy), agent2.policy, softmax(agent2.policy), results)

            if(args.use_softmax):
                history[i,0:2] = history[i,0:2] + softmax(agent1.policy)
                history[i,2:4] = history[i,2:4] + softmax(agent2.policy)
            else:
                history[i,0:2] = history[i,0:2] + agent1.policy/agent1.policy.sum()
                history[i,2:4] = history[i,2:4] + agent2.policy/agent2.policy.sum()

    f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plotPolicyOverTime(ax1, history[:,0:4]/args.N, pris.POLICY_LEGEND, args)
    plotTimeAveragedPolicy(ax2, history[:,0:4]/args.N, pris.POLICY_LEGEND, args)
    plt.show()


if __name__ == '__main__':
    COMMAND_MAP = {'rps':rock_paper_scissors, 'prisoner':prisoners_dilema}

    parser = argparse.ArgumentParser(description="Play multiagent games")
    parser.add_argument('game', default='rps', choices=COMMAND_MAP.keys(), help="You must pass one of the following games to run: " + str(COMMAND_MAP.keys()))
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int, default=1000, help='Sets the number of steps to run the simulation or the number of episodes to run if the game is episodic.')
    parser.add_argument('--off-policy', dest='off_policy', action='store_true', help='If set to true will run all agents with a balanced random policy.')
    parser.add_argument('-l', '--learn-rate', dest='learning_rate', type=float, default=0.01, help='Set the learning rate used in policy optimization.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enables verbose printing while simulation. This degrades performance considerably.')
    parser.add_argument('--random-start', dest='random_start', action='store_true', help='Randomly initialize policies between the parameters given by --rs-max and --rs-min')
    parser.add_argument('--rs-max', dest='random_start_max', default=1.0, type=float, help='Upper bound to use during random initialization.')
    parser.add_argument('--rs-min', dest='random_start_min', default=-1.0, type=float, help='Lower bound to use during random initialization.')
    parser.add_argument('--optimizer', dest='optimizer', choices=['td', 'multi-w'], default='multi-w')
    parser.add_argument('-N', dest='N', default=1, type=int, help='Average data over N runs of the simulation.')
    parser.add_argument('--use-softmax', dest='use_softmax', action='store_true', help='Setting this to true will cause the agent\'s to use softmax for action selection.')
    args = parser.parse_args()

    # get the game to run.
    game = COMMAND_MAP[args.game]
    #run it.
    game(args)
