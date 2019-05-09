
import argparse
import rps as rps
import numpy as np
from optimizers import PolcyIncrementOptimizer

def rock_paper_scissors(args):
    agent1 = rps.RPSAgent(PolcyIncrementOptimizer(0.8))
    agent2 = rps.RPSAgent(PolcyIncrementOptimizer(0.8))
    game = rps.RPSGame([agent1, agent2])

    for i in range(args.n_iterations):
        results = game.update()
        print(i, agent1.getNormalizedPolicy(), agent2.getNormalizedPolicy(), results)

if __name__ == '__main__':
    COMMAND_MAP = {'rps':rock_paper_scissors}

    parser = argparse.ArgumentParser(description="Play multiagent games")
    parser.add_argument('game', default='rps', choices=COMMAND_MAP.keys(), help="You must pass one of the following games to run: " + str(COMMAND_MAP.keys()))
    parser.add_argument('-i', '--iterations', dest='n_iterations', type=int, default=1000)
    args = parser.parse_args()

    # get the game to run.
    game = COMMAND_MAP[args.game]
    #run it.
    game(args)
