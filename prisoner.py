
import numpy as np

COOPERATE = 0
DEFECT = 1

#           COOPERATE   DEFECT
#COOPERATE  -1          -10
#DEFECT      0          -4

PAYOUT_MATRIX = np.array([[-0.1, -1.0],
                          [0,  -0.4]])