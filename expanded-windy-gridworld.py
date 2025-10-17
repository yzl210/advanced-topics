from collections import defaultdict

import matplotlib.pyplot as plt
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


def epsilon_greedy(Q, s, nA, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(nA))
    r, c = s
    row = Q[r, c]
    best = np.flatnonzero(row == row.max())
    return int(rng.choice(best))


def sarsa(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=300, seed=0, q_init=0.0, max_steps=300):
    rng = np.random.default_rng(seed)
    Q = np.full((env.H, env.W, env.nA), q_init, dtype=float)
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
        while steps < max_steps:
            S_prime, R, terminal = env.step(A)
            if terminal:
                rS, cS = S
                Q[rS, cS, A] += alpha * (R - Q[rS, cS, A])
                steps += 1
                break
            A_prime = policy(S_prime)
            rS, cS = S
            rSp, cSp = S_prime
            target = R + gamma * Q[rSp, cSp, A_prime]
            Q[rS, cS, A] += alpha * (target - Q[rS, cS, A])
            S, A = S_prime, A_prime
            steps += 1
        steps_per_episode.append(steps)
    return steps_per_episode


def q_learning(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=300, seed=1, q_init=0.0, max_steps=300):
    rng = np.random.default_rng(seed)
    Q = np.full((env.H, env.W, env.nA), q_init, dtype=float)
    steps = []
    for _ in range(episodes):
        gr, gc = env.goal
        Q[gr, gc, :] = 0.0
        s = env.reset()
        t = 0
        while t < max_steps:
            a = epsilon_greedy(Q, s, env.nA, epsilon, rng)
            s2, r, done = env.step(a)
            rS, cS = s
            if done:
                Q[rS, cS, a] += alpha * (r - Q[rS, cS, a])
                t += 1
                break
            rSp, cSp = s2
            target = r + gamma * np.max(Q[rSp, cSp])
            Q[rS, cS, a] += alpha * (target - Q[rS, cS, a])
            s = s2
            t += 1
        steps.append(t)
    return steps


def double_q_learning(env, alpha=0.5, gamma=1.0, epsilon=0.1, episodes=300, seed=2, q_init=0.0, max_steps=300):
    rng = np.random.default_rng(seed)
    Qa = np.full((env.H, env.W, env.nA), q_init, dtype=float)
    Qb = np.full((env.H, env.W, env.nA), q_init, dtype=float)
    steps = []
    for _ in range(episodes):
        gr, gc = env.goal
        Qa[gr, gc, :] = 0.0
        Qb[gr, gc, :] = 0.0
        s = env.reset()
        t = 0
        while t < max_steps:
            Qsum = Qa + Qb
            a = epsilon_greedy(Qsum, s, env.nA, epsilon, rng)
            s2, r, done = env.step(a)
            rS, cS = s
            rSp, cSp = s2
            if done:
                if rng.random() < 0.5:
                    Qa[rS, cS, a] += alpha * (r - Qa[rS, cS, a])
                else:
                    Qb[rS, cS, a] += alpha * (r - Qb[rS, cS, a])
                t += 1
                break
            if rng.random() < 0.5:
                a_star = np.argmax(Qa[rSp, cSp])
                target = r + gamma * Qb[rSp, cSp, a_star]
                Qa[rS, cS, a] += alpha * (target - Qa[rS, cS, a])
            else:
                b_star = np.argmax(Qb[rSp, cSp])
                target = r + gamma * Qa[rSp, cSp, b_star]
                Qb[rS, cS, a] += alpha * (target - Qb[rS, cS, a])
            s = s2
            t += 1
        steps.append(t)
    return steps


def monte_carlo(env, gamma=1.0, epsilon=0.1, episodes=600, seed=3, q_init=0.0, max_steps=400):
    rng = np.random.default_rng(seed)
    Q = np.full((env.H, env.W, env.nA), q_init, dtype=float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    steps = []

    def gen_episode():
        s = env.reset()
        t = 0
        episode = []
        while t < max_steps:
            a = epsilon_greedy(Q, s, env.nA, epsilon, rng)
            s2, r, done = env.step(a)
            episode.append((s, a, r))
            s = s2
            t += 1
            if done:
                break
        return episode, t

    for _ in range(episodes):
        gr, gc = env.goal
        Q[gr, gc, :] = 0.0
        ep, t = gen_episode()
        steps.append(t)
        G = 0.0
        visited_sa = set()
        for s, a, r in reversed(ep):
            G = gamma * G + r
            key = (s, a)
            if key not in visited_sa:
                returns_sum[key] += G
                returns_count[key] += 1
                rS, cS = s
                Q[rS, cS, a] = returns_sum[key] / max(1, returns_count[key])
                visited_sa.add(key)
    return steps


def moving_average(x, w=20):
    if len(x) < w:
        return np.array(x, dtype=float)
    c = np.cumsum(np.insert(np.array(x, dtype=float), 0, 0))
    return (c[w:] - c[:-w]) / float(w)


def plot(env, title, episodes_td=600, episodes_mc=600, seed=0):
    curves = {}
    curves["SARSA"] = sarsa(env, episodes=episodes_td, seed=seed + 0, q_init=5.0, alpha=0.5, epsilon=0.1, gamma=1.0,
                            max_steps=500)
    curves["Q-Learning"] = q_learning(env, episodes=episodes_td, seed=seed + 1, q_init=2.0, alpha=0.5, epsilon=0.1,
                                      gamma=1.0, max_steps=500)
    curves["Double Q"] = double_q_learning(env, episodes=episodes_td, seed=seed + 2, q_init=2.0, alpha=0.5, epsilon=0.1,
                                           gamma=1.0, max_steps=500)
    curves["Monte Carlo"] = monte_carlo(env, episodes=episodes_mc, seed=seed + 3, q_init=0.0, epsilon=0.1, gamma=1.0,
                                        max_steps=500)

    plt.figure()
    for name, y in curves.items():
        x = np.arange(len(y))
        w = 20 if name != "Monte Carlo" else 35
        y_sm = moving_average(y, w=w)
        x_sm = x[len(x) - len(y_sm):]
        plt.plot(x_sm, y_sm, label=name)
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")
    plt.title(f"Learning Curves - {title}")
    plt.legend()
    plt.tight_layout()
    plt.show()


env4 = WindyGridworld(kings_moves=False, no_op=False)
env8 = WindyGridworld(kings_moves=True, no_op=False)
env9 = WindyGridworld(kings_moves=True, no_op=True)

plot(env4, "4-action", seed=0)
plot(env8, "8-action (kings moves)", seed=10)
plot(env9, "9-action (kings moves + no-op)", seed=20)
