import numpy as np
from optimizers import PolcyIncrementOptimizer, MultiplicitiveWeightOptimizer

def initiallzePolicy(n_actions, args):
    if(args.random_start):
        return np.random.rand(n_actions)*abs(args.random_start_max - args.random_start_min) + args.random_start_min

    return np.zeros(n_actions) 

def initializeOptimizer(args):
    if args.optimizer == 'multi-w':
        return MultiplicitiveWeightOptimizer(args.learning_rate)
    elif args.optimizer == 'td':
        return PolcyIncrementOptimizer(args.learning_rate)

    return None