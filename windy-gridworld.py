import numpy as np


class WindyGridworld:
    def __init__(self, height=7, width=10, start=(3, 0), goal=(3, 7), wind=(0, 0, 0, 1, 1, 1, 2, 2, 1, 0),
                 kings_moves=False, no_op=False):
        self.H = height
        self.W = width
        self.start = start
        self.goal = goal
        self.wind = np.array(wind)
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        if kings_moves:
            self.actions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if no_op:
            self.actions.append((0, 0))
        self.nA = len(self.actions)

    def reset(self):
        self.r, self.c = self.start
        return self.r, self.c

    def step(self, a):
        dr, dc = self.actions[a]
        r, c = self.r, self.c
        r = np.clip(r + dr, 0, self.H - 1)
        c = np.clip(c + dc, 0, self.W - 1)
        r = np.clip(r - self.wind[c], 0, self.H - 1)
        self.r, self.c = r, c
        done = (r, c) == self.goal
        return (r, c), -1, done


def sarsa(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=200, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.H, env.W, env.nA), dtype=float)
    steps_per_episode = []

    def policy(s):
        if rng.random() < epsilon:
            return int(rng.integers(env.nA))
        r, c = s
        row = Q[r, c]
        best = np.flatnonzero(row == row.max())
        return int(rng.choice(best))

    for _ in range(episodes):
        gr, gc = env.goal
        Q[gr, gc, :] = 0.0

        S = env.reset()
        A = policy(S)
        steps = 0

        while True:
            S_prime, R, terminal = env.step(A)
            if terminal:
                rS, cS = S
                Q[rS, cS, A] += alpha * (R - Q[rS, cS, A])
                steps += 1
                steps_per_episode.append(steps)
                break

            A_prime = policy(S_prime)

            rS, cS = S
            rSp, cSp = S_prime
            target = R + gamma * Q[rSp, cSp, A_prime]
            Q[rS, cS, A] += alpha * (target - Q[rS, cS, A])

            S, A = S_prime, A_prime
            steps += 1

    return Q, np.array(steps_per_episode)


def print_optimal_path(Q, env):
    s = env.reset()
    path = [s]
    visited = set([s])

    while s != env.goal:
        r, c = s
        A = np.argmax(Q[r, c])
        s, _, _ = env.step(A)

        if s in visited:
            print("Loop detected, stopping.")
            break
        visited.add(s)
        path.append(s)

        if len(path) > env.H * env.W:
            print("Path too long, stopping.")
            break

    grid = [["." for _ in range(env.W)] for _ in range(env.H)]
    for (r, c) in path:
        grid[r][c] = "*"
    sr, sc = env.start
    gr, gc = env.goal
    grid[sr][sc] = "S"
    grid[gr][gc] = "G"

    for row in grid:
        print(" ".join(row))


env4 = WindyGridworld(kings_moves=False, no_op=False)
env8 = WindyGridworld(kings_moves=True, no_op=False)
env9 = WindyGridworld(kings_moves=True, no_op=True)

Q4, steps4 = sarsa(env4, episodes=1000)
Q8, steps8 = sarsa(env8, episodes=1000)
Q9, steps9 = sarsa(env9, episodes=1000)

print("Optimal path for 4-action windy gridworld:")
print_optimal_path(Q4, env4)
print("Optimal path for windy gridworld with kings moves:")
print_optimal_path(Q8, env8)
print("Optimal path for windy gridworld with kings moves and no-op:")
print_optimal_path(Q9, env9)
