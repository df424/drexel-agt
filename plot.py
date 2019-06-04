import numpy as np
import matplotlib.pyplot as plt

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