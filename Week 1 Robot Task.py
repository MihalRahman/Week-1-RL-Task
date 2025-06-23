import numpy as np

#Parameters
GRID_SIZE   = 5
P_SUCCESS   = 0.9
P_SLIP      = (1 - P_SUCCESS) / 3.0 #accidentally moving in the wrong direction, to introduce randomness
BETA        = 0.9          
THRESHOLD   = 1e-4 #for value iteration convergence
STEP_COST   = -1 #cost to move

#Build the zig-zag base reward grid
reward_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

#Fill with zig-zag 1-25
counter = 1
for row in reversed(range(GRID_SIZE)): #start at bottom row (row index 4) up to 0
    if (GRID_SIZE - 1 - row) % 2 == 0:
        #bottom row: right->left
        cols = reversed(range(GRID_SIZE))
    else:
        #next row up: left->right
        cols = range(GRID_SIZE)
    for col in cols:
        reward_grid[row, col] = counter
        counter += 1

#Override corners
reward_grid[4, 4] = +1 #start
reward_grid[0, 0] = +100 #goal

#Subtract 30 in the central 3×3 grid
for i in range(1, GRID_SIZE-1):
    for j in range(1, GRID_SIZE-1):
        reward_grid[i, j] -= 30

#MDP
actions = {
    'U': (-1,  0),
    'D': (+1,  0),
    'L': ( 0, -1),
    'R': ( 0, +1),
}
action_list = list(actions.keys())

def in_bounds(r, c):
    return 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE

#Calculate Transition probabilities:
#P[next_state | (state, action)]
#and immediate reward R(s,a,s')
transitions = {}
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        s = (r, c)
        transitions[s] = {}
        if s == (0, 0):
            continue  #terminal state
        for a in action_list:
            probs = []
            for a2 in action_list:
                prob = P_SUCCESS if (a2 == a) else P_SLIP
                dr, dc = actions[a2]
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc):
                    nr, nc = r, c  #hit wall and stay
                # reward when landing in (nr,nc), plus cost to move:
                rwd = reward_grid[nr, nc] + STEP_COST
                probs.append((prob, (nr, nc), rwd))
            transitions[s][a] = probs

#Value iteration
V = np.zeros((GRID_SIZE, GRID_SIZE))
while True:
    delta = 0
    V_new = V.copy()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            s = (r, c)
            if s == (0, 0):
                continue
            q_vals = []
            for a in action_list:
                q_sa = sum(
                    prob * (rwd + BETA * V[next_s])
                    for prob, next_s, rwd in transitions[s][a]
                )
                q_vals.append(q_sa)
            best_q = max(q_vals)
            V_new[r, c] = best_q
            delta = max(delta, abs(V_new[r, c] - V[r, c]))
    V = V_new
    if delta < THRESHOLD:
        break

#Extract greedy policy
policy = np.full((GRID_SIZE, GRID_SIZE), ' ', dtype='<U1')
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        s = (r, c)
        if s == (0,0):
            policy[r, c] = 'G'  # goal
        else:
            # pick action with highest expected value
            best_a, best_val = None, -1e9
            for a in action_list:
                q_sa = sum(
                    prob * (rwd + BETA * V[next_s])
                    for prob, next_s, rwd in transitions[s][a]
                )
                if q_sa > best_val:
                    best_val, best_a = q_sa, a
            policy[r, c] = best_a

#Simulate one greedy run from start
path = []
rewards = []
current = (4, 4)
total_reward = 0.0
max_steps = 100
for _ in range(max_steps):
    path.append(current)
    if current == (0,0):
        break
    a = policy[current]
    # sample deterministically the most likely next state (for illustration)
    # actually we would follow the intended action:
    dr, dc = actions[a]
    nr, nc = current[0] + dr, current[1] + dc
    if not in_bounds(nr, nc):
        nr, nc = current
    rwd = reward_grid[nr, nc] + STEP_COST
    total_reward += rwd
    rewards.append(rwd)
    current = (nr, nc)

#Output
print("Reward grid:\n", reward_grid, "\n")
print("Optimal value function V*:\n", np.round(V,1), "\n")
print("Optimal policy (U/D/L/R):\n", policy, "\n")
print("Greedy path from start to goal:", path)
print("Cumulative reward along that path: {:.2f}".format(total_reward))
