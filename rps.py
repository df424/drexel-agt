
import numpy as np

ROCK = 0
PAPER = 1
SCISSORS = 2

#           Rock        Paper       Scissors
#Rock       0           -1          1
#Paper      1           0           -1
#Scissors   -1          1           0

PAYOUT_MATRIX = np.array([[0, -1,  1],
                          [1,  0, -1],
                          [-1, 1,  0]])

POLICY_LEGEND =  ['agent1 rock', 'agent1 paper', 'agent1 scissors', 'agent2 rock', 'agent2 paper', 'agent2 scissors']