import numpy as np
import matplotlib.pyplot as plt

def plotPolicyOverTime(axis, history, legend, args):
    indices = np.arange(0, args.n_iterations)

    # for each of our agents...
    for h in range(len(history)):
        a_h = history[h]

        # compute the average policy across runs...
        averaged_samples = a_h['policy'].mean(axis=0)

        # for each action...
        for i in range(averaged_samples.shape[1]):
            # plot the data as error bars.
            axis.errorbar(indices, y=averaged_samples[:,i], yerr=np.abs(a_h['policy'].std(axis=0)[:,i]), errorevery=max(1, args.n_iterations/50))

    # format the graph.
    axis.set_title('Policy vs. Iterations, λ=' + str(args.learning_rate) + ', N=' + str(args.N))
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Normalized Policy')
    axis.legend(legend)

def plotTimeAveragedPolicy(axis, history, legend, args):
    indices = np.arange(0, args.n_iterations)

    # for each agent...
    for i in range(len(history)):
        actor_history = history[i]
        policy_history = actor_history['policy']

        # compute the cumulative sums for the policies for each iteration.
        cumulative_sums = np.cumsum(policy_history, axis=1)

        # make them time averaged..
        time_averaged_sums = cumulative_sums/(indices+1).reshape(args.n_iterations,1)
        
        # calculate the standard deviation across runs.
        std = time_averaged_sums.std(axis=0)

        # for each agent and each action that it could take...
        for i in range(policy_history.shape[2]):
            # plot the data as error bars.
            axis.errorbar(indices, y=time_averaged_sums.mean(axis=0)[:,i], yerr=std[:,i], errorevery=max(1,args.n_iterations/50))

    # format the graph.
    axis.set_title('Time Averaged Policies, λ=' + str(args.learning_rate) + ', N=' + str(args.N))
    axis.set_xlabel('Iterations')
    axis.set_ylabel('Time Averaged Policy')
    axis.grid()
    axis.legend(legend)

def plotRegret(axis, history, legend, args):
    # for each agent...
    for i in range(len(history)):
        action_history = history[i]['actions']
        reward_history = history[i]['rewards']

        print(action_history)
        print(reward_history)

        # rewards is an N x i x |A| matrix summing over i for each N will give us all of the best fixed action sequences if we take the argmax...
        best_fixed_actions = np.argmax(reward_history.mean(axis=1), axis=1)

        # now for each run we can cumulative sum over the best fixed action sequence which is the first term of our regret...
        best_fixed_sums = np.cumsum(reward_history[np.arange(len(reward_history)), :, best_fixed_actions], axis=1)

        # compute the cumulative sum of the actual actions as well...
        print(action_history)
        print(reward_history.shape, best_fixed_actions)
        print(reward_history[np.arange(len(reward_history)),np.arange(len(reward_history[0])),best_fixed_actions])
        print(best_fixed_sums)

        import sys
        sys.exit()
