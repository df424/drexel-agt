
import argparse
import rps as rps
import numpy as np
import simple_two_player as stpg
import matplotlib.pyplot as plt
from optimizers import PolcyIncrementOptimizer

def rock_paper_scissors(args):
    agent1 = stpg.VectorPolicyAgent(PolcyIncrementOptimizer(args.learning_rate), args.off_policy, np.random.rand(3)*2-1)
    agent2 = stpg.VectorPolicyAgent(PolcyIncrementOptimizer(args.learning_rate), args.off_policy, np.random.rand(3)*2-1)
    game = stpg.EpisodicGame([agent1, agent2], rps.PAYOUT_MATRIX)

    policy_plots = np.zeros((args.n_iterations,6))

    for i in range(args.n_iterations):
        #update the game
        results = game.update()

        # print if verbosity is on.
        if(args.verbose):
            print(i, agent1.policy, agent1.getNormalizedPolicy(), agent2.policy, agent2.getNormalizedPolicy(), results)

        policy_plots[i,0:3] = agent1.policy
        policy_plots[i,3:6] = agent2.policy
        #policy_plots[i,0:3] = agent1.getNormalizedPolicy()
        #policy_plots[i,3:6] = agent2.getNormalizedPolicy()

    plt.plot(policy_plots[:]) 
    plt.legend(['agent1 rock', 'agent1 paper', 'agent1 scissors', 'agent2 rock', 'agent2 paper', 'agent2 scissors'])
    plt.show()

if __name__ == '__main__':
    COMMAND_MAP = {'rps':rock_paper_scissors}

    parser = argparse.ArgumentParser(description="Play multiagent games")
    parser.add_argument('game', default='rps', choices=COMMAND_MAP.keys(), help="You must pass one of the following games to run: " + str(COMMAND_MAP.keys()))
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int, default=1000)
    parser.add_argument('--off-policy', dest='off_policy', action='store_true')
    parser.add_argument('-l', '--learn-rate', dest='learning_rate', type=float, default=0.01)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    args = parser.parse_args()

    # get the game to run.
    game = COMMAND_MAP[args.game]
    #run it.
    game(args)
