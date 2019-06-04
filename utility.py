import numpy as np
import logging
from optimizers import PolcyIncrementOptimizer, MultiplicitiveWeightOptimizer

def initiallzePolicy(n_actions, args):
    if(args.random_start):
        ran = np.random.rand(n_actions)
        range = abs(args.random_start_max - args.random_start_min) + args.random_start_min
        logging.debug(ran, range)
        return ran*range

    return np.zeros(n_actions) 

def initializeOptimizer(args):
    if args.optimizer == 'multi-w':
        return MultiplicitiveWeightOptimizer(args.learning_rate)
    elif args.optimizer == 'td':
        return PolcyIncrementOptimizer(args.learning_rate)

    return None