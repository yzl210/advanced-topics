import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

MAX_CARS = 20
MAX_MOVE = 5
RENT_REWARD = 10
MOVE_COST_PER_CAR = 2
PARKING_THRESHOLD = 10
PARKING_PENALTY = 4
GAMMA = 0.9

def poisson_table(lmbda, N=8):
    xs = np.arange(0, N + 1)
    pmf = np.array([math.exp(-lmbda) * (lmbda ** x) / math.factorial(x) for x in xs])
    pmf /= pmf.sum()
    return pmf

poisson_req1 = poisson_table(3)
poisson_req2 = poisson_table(4)
poisson_ret1 = poisson_table(3)
poisson_ret2 = poisson_table(2)

def feasible_actions(i, j):
    lo = max(-min(MAX_MOVE, j), -(MAX_CARS - i))
    hi = min(min(MAX_MOVE, i), MAX_CARS - j)
    return list(range(lo, hi + 1))

def move_cost(a):
    return MOVE_COST_PER_CAR * max(0, a - 1) if a > 0 else MOVE_COST_PER_CAR * (-a)

def parking_cost(i_after, j_after):
    cost = 0
    if i_after > PARKING_THRESHOLD:
        cost += PARKING_PENALTY
    if j_after > PARKING_THRESHOLD:
        cost += PARKING_PENALTY
    return cost

transitions = defaultdict(list)
states = [(i,j) for i in range(MAX_CARS+1) for j in range(MAX_CARS+1)]
total_entries = 0

for i in range(MAX_CARS+1):
    for j in range(MAX_CARS+1):
        acts = feasible_actions(i, j)
        for action in acts:
            i_after = min(MAX_CARS, max(0, i - action))
            j_after = min(MAX_CARS, max(0, j + action))
            base_reward = -(move_cost(action) + parking_cost(i_after, j_after))

            probs = np.zeros((MAX_CARS+1, MAX_CARS+1), dtype=float)
            rewards_add = np.zeros_like(probs)

            for req1, p_r1 in enumerate(poisson_req1):
                rent1 = min(i_after, req1)
                remain1 = i_after - rent1
                rev1 = RENT_REWARD * rent1

                for req2, p_r2 in enumerate(poisson_req2):
                    rent2 = min(j_after, req2)
                    remain2 = j_after - rent2
                    rev2 = RENT_REWARD * rent2
                    prob_req = p_r1 * p_r2
                    immediate = base_reward + rev1 + rev2

                    for ret1, p_t1 in enumerate(poisson_ret1):
                        n1 = min(MAX_CARS, remain1 + ret1)
                        for ret2, p_t2 in enumerate(poisson_ret2):
                            n2 = min(MAX_CARS, remain2 + ret2)
                            p = prob_req * p_t1 * p_t2
                            probs[n1, n2] += p
                            rewards_add[n1, n2] += p * immediate

            entries = np.argwhere(probs > 0)
            for (n1, n2) in entries:
                transitions[(i, j, action)].append((probs[n1, n2], n1, n2, rewards_add[n1, n2]))
            total_entries += len(entries)

V = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=float)
policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1), dtype=int)
theta = 1e-3

def value_of(i, j, a, V):
    s = 0.0
    for p, n1, n2, rew in transitions[(i, j, a)]:
        s += rew + p * GAMMA * V[n1, n2]
    return s

policy_stable = False
improve_steps = 0
while not policy_stable:
    improve_steps += 1
    while True:
        delta = 0.0
        for i in range(MAX_CARS + 1):
            for j in range(MAX_CARS + 1):
                value = V[i, j]
                action = policy[i, j]
                V[i, j] = value_of(i, j, action, V)
                delta = max(delta, abs(value - V[i, j]))
        if delta < theta:
            break

    policy_stable = True
    for i in range(MAX_CARS + 1):
        for j in range(MAX_CARS + 1):
            old_action = policy[i, j]
            best_value = -1e18
            best_action = old_action
            for action in feasible_actions(i, j):
                val = value_of(i, j, action, V)
                if val > best_value + 1e-9:
                    best_value = val
                    best_action = action
            policy[i, j] = best_action
            if best_action != old_action:
                policy_stable = False

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(policy.T, origin='lower', extent=[0, MAX_CARS, 0, MAX_CARS], interpolation='nearest')
ax.set_xlabel('# Cars at first location')
ax.set_ylabel('# Cars at second location')
ax.set_title('Optimal Policy')
plt.colorbar(im, ax=ax, label='Action a')
plt.show()
