import numpy as np

import random

l = [[1,2],[3,4],[5,6]]
q_per_state = np.zeros((2,3))
for action in range(len(l)):
    for state in range(len(l[action])):
        q_per_state[state][action] = l[action][state]
print(q_per_state)
    
