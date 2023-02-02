"""
    x/100       45
  _________B_________
 /                   \ 
A                     C
 \_________D_________/
    45        X/100

the edges are listed in that order :
A -> B
B -> C
A -> D
B -> D
D -> B
D -> C
"""
from scipy.optimize import linprog
import numpy as np

threshold = 50  # threshold condition for stopping the algorithm

# A[i, j] is set to +1 / -1 / 0 if people from edge j come to / depart from / do not go to vertex i
A = np.array(
    [
        [-1,  0, -1,  0, ],
        [1, -1,  0,  0, ],
        [0,  0,  1, -1, ],
        [0,  1,  0,  1, ],
    ]
)

# -4000 go from vertex A, +4000 arrive to vertex C
b = np.array([-4000, 0, 0, +4000])

# Compute the cost vector from a given flow


def cost_function(flow):
    return np.array([flow[0]/100, 45, 45, flow[3]/100])


flow = np.array([4000, 4000, 0, 0])  # flow for iteration 0 : A -> B -> C
last_flow = np.zeros(flow.shape)

i = 0  # iteration counter

while np.sum(np.abs(flow - last_flow)) > threshold:  # || flow - last flow ||1 > threshold

    # Compute step
    step = 1 / (i + 2)

    # Update the cost for each edge
    costs = cost_function(flow)

    last_flow = flow

    # Solve the linear problem
    next_flow = linprog(costs, A_eq=A, b_eq=b,
                        options={"rr": False, })['x']

    # Compute the next flow
    flow = (1 - step) * last_flow + step * next_flow

    i += 1

print("Last flow :", flow)
