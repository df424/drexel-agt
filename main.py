
import argparse
import logging
import rps as rps
import prisoner as pris
import numpy as np
import matplotlib.pyplot as plt
import math
from optimizers import PolcyIncrementOptimizer, MultiplicitiveWeightOptimizer
from scipy.special import softmax

def plotPolicyOverTime(axis, policy_history, legend, args):
    indices = np.arange(0, args.n_iterations)
    # compute the average policy across runs...
    averaged_samples = policy_history.mean(axis=0)

    # for each agent and each action that it could take...
    for i in range(policy_history.shape[2]):
        # plot the data as error bars.
        axis.errorbar(indices, y=averaged_samples[:,i], yerr=policy_history.std(axis=0)[:,i], errorevery=args.n_iterations/50)

    # format the graph.
    axis.set_title('Policy vs. Iterations, λ=' + str(args.learning_rate) + ', N=' + str(args.N))
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Normalized Policy')
    axis.legend(legend)

def plotTimeAveragedPolicy(axis, policy_history, legend, args):
    indices = np.arange(0, args.n_iterations)
    # compute the cumulative sums for the policies for each iteration.
    cumulative_sums = np.cumsum(policy_history, axis=1)
    # make them time averaged..
    time_averaged_sums = cumulative_sums/(indices+1).reshape(args.n_iterations,1)
    # calculate the standard deviation across runs.
    std = time_averaged_sums.std(axis=0)

    # for each agent and each action that it could take...
    for i in range(policy_history.shape[2]):
        # plot the data as error bars.
        axis.errorbar(indices, y=time_averaged_sums.mean(axis=0)[:,i], yerr=std[:,i], errorevery=args.n_iterations/50)

    # format the graph.
    axis.set_title('Time Averaged Policies, λ=' + str(args.learning_rate) + ', N=' + str(args.N))
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Time Averaged Policy')
    axis.grid()
    axis.legend(legend)

def rock_paper_scissors(args):
    game = rps.RPS(args)
    history = np.zeros((args.N, args.n_iterations,6))
    for n in range(args.N):
        print(str(round(n*100.0/args.N)) + '%...')
        # Reset the agents initial policy...
        game.reset()
        for i in range(args.n_iterations):
            # update the game
            results = game.update()

            # print if verbosity is on.
            if(args.verbose):
                print(i)
                for actor in game.actors:
                    print(actor.policy, softmax(actor.policy))
                print(results)

            if(args.use_softmax):
                history[n, i,0:3] = softmax(game.actors[0].policy)
                history[n, i,3:6] = softmax(game.actors[1].policy)
            else:
                history[n, i,0:3] = game.actors[0].policy/game.actors[0].policy.sum()
                history[n, i,3:6] = game.actors[1].policy/game.actors[1].policy.sum()

    f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plotPolicyOverTime(ax1, history[:,:,0:6], rps.POLICY_LEGEND, args)    
    plotTimeAveragedPolicy(ax2, history[:,:,0:6], rps.POLICY_LEGEND, args)
    plt.show()

def prisoners_dilema(args):
    game = pris.Prisoner(args)

    history = np.zeros((args.N, args.n_iterations,4))

    for n in range(args.N):
        print(str(round(n*100.0/args.N)) + '%...')
        # Reset the agents initial policy...
        game.reset()
        for i in range(args.n_iterations):
            #update the game
            results = game.update()

            # print if verbosity is on.
            if(args.verbose):
                print(i)
                for actor in game.actors:
                    print(actor.policy, softmax(actor.policy))
                print(results)

            if(args.use_softmax):
                history[n,i,0:2] = softmax(game.actors[0].policy)
                history[n,i,2:4] = softmax(game.actors[1].policy)
            else:
                history[n,i,0:2] = game.actors[0].policy/game.actors[0].policy.sum()
                history[n,i,2:4] = game.actors[1].policy/game.actors[1].policy.sum()

    f,(ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plotPolicyOverTime(ax1, history[:,:,0:4], pris.POLICY_LEGEND, args)
    plotTimeAveragedPolicy(ax2, history[:,:,0:4], pris.POLICY_LEGEND, args)
    plt.show()


if __name__ == '__main__':
    COMMAND_MAP = {'rps':rock_paper_scissors, 'prisoner':prisoners_dilema}

    parser = argparse.ArgumentParser(description="Play multiagent games")
    parser.add_argument('game', default='rps', choices=COMMAND_MAP.keys(), help="You must pass one of the following games to run: " + str(COMMAND_MAP.keys()))
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int, default=10000, help='Sets the number of steps to run the simulation or the number of episodes to run if the game is episodic.')
    parser.add_argument('--off-policy', dest='off_policy', action='store_true', help='If set to true will run all agents with a balanced random policy.')
    parser.add_argument('-l', '--learn-rate', dest='learning_rate', type=float, default=0.01, help='Set the learning rate used in policy optimization.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enables verbose printing while simulation. This degrades performance considerably.')
    parser.add_argument('--random-start', dest='random_start', action='store_true', help='Randomly initialize policies between the parameters given by --rs-max and --rs-min')
    parser.add_argument('--rs-max', dest='random_start_max', default=1.0, type=float, help='Upper bound to use during random initialization.')
    parser.add_argument('--rs-min', dest='random_start_min', default=0, type=float, help='Lower bound to use during random initialization.')
    parser.add_argument('--optimizer', dest='optimizer', choices=['td', 'multi-w'], default='multi-w')
    parser.add_argument('-N', dest='N', default=1, type=int, help='Average data over N runs of the simulation.')
    parser.add_argument('--use-softmax', dest='use_softmax', action='store_true', help='Setting this to true will cause the agent\'s to use softmax for action selection.')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # get the game to run.
    game = COMMAND_MAP[args.game]
    #run it.
    game(args)
