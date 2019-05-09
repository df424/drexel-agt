
import numpy as np

COOPERATE = 0
DEFECT = 1

#           COOPERATE   DEFECT
#COOPERATE  -1          -10
#DEFECT      0          -4

PAYOUT_MATRIX = np.array([[-1, -10],
                          [0,  -4]])