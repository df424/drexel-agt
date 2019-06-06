
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from optimizers import PolcyIncrementOptimizer, MultiplicitiveWeightOptimizer
from rps import RPS
from chicken import Chicken
from prisoner import Prisoner

COMMAND_MAP = {
        'rps':RPS,
        'prisoner':Prisoner,
        'chicken':Chicken
    }

def main():
    args = parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # get the game to run.
    game_name = COMMAND_MAP[args.game]
    # run it.
    game = game_name(args)
    game.initHistory(args.N, args.n_iterations)
    for n in range(args.N):
        print(str(round(n*100.0/args.N)) + '%...')
        # Reset the agents initial policy...
        game.reset()
        for i in range(args.n_iterations):
            # update the game
            actions, rewards = game.update()
            game.updateHistory(n, i, actions, rewards, args.use_softmax)

            # print if verbosity is on.
            # if(args.verbose):
            #     print(i)
            #     for actor in game.actors:
            #         print(actor.policy, softmax(actor.policy))
            #     print(results)
    print('100%')
    game.display(args)


def parse_args():
    parser = argparse.ArgumentParser(description="Play multiagent games")
    parser.add_argument('game', default='rps', choices=COMMAND_MAP.keys(), help="You must pass one of the following games to run: " + str(COMMAND_MAP.keys()))
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int, default=10000, help='Sets the number of steps to run the simulation or the number of episodes to run if the game is episodic.')
    parser.add_argument('--off-policy', dest='off_policy', action='store_true', help='If set to true will run all agents with a balanced random policy.')
    parser.add_argument('-l', '--learn-rate', dest='learning_rate', type=float, default=0.01, help='Set the learning rate used in policy optimization.')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enables verbose printing while simulation. This degrades performance considerably.')
    parser.add_argument('--random-start', dest='random_start', action='store_false', help='Randomly initialize policies between the parameters given by --rs-max and --rs-min')
    parser.add_argument('--rs-max', dest='random_start_max', default=1.0, type=float, help='Upper bound to use during random initialization.')
    parser.add_argument('--rs-min', dest='random_start_min', default=0, type=float, help='Lower bound to use during random initialization.')
    parser.add_argument('--optimizer', dest='optimizer', choices=['td', 'multi-w'], default='multi-w')
    parser.add_argument('-N', dest='N', default=1, type=int, help='Average data over N runs of the simulation.')
    parser.add_argument('--use-softmax', dest='use_softmax', action='store_true', help='Setting this to true will cause the agent\'s to use softmax for action selection.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
