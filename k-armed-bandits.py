import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter


class Bandit:
    def __init__(self, num_arms, initial_radius=0):
        self.num_arms = num_arms
        self.arm_means = np.random.uniform(-initial_radius, initial_radius, self.num_arms)

    def pull_arm(self, arm: int):
        return np.random.normal(self.arm_means[arm], 1)

    def shuffle(self, sd=0.01):
        self.arm_means += np.random.normal(0, sd, self.num_arms)

    def optimal(self):
        return np.argmax(self.arm_means)


class Agent:
    def __init__(self, num_arms, epsilon=0.1, alpha=None):
        self.epsilon = epsilon
        self.alpha = alpha
        self.q = np.zeros(num_arms)
        self.n = np.zeros(num_arms)
        self.rewards = [[] for _ in range(num_arms)]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.q))
        return np.random.choice(np.flatnonzero(self.q == self.q.max()))

    def update(self, action, reward):
        self.n[action] += 1
        self.rewards[action].append(reward)

        if self.alpha is None:
            self.q[action] = np.mean(self.rewards[action])
        else:
            if self.n[action] == 1:
                self.q[action] = reward
            else:
                self.q[action] += self.alpha * (reward - self.q[action])


def run(runs=50, steps=10000):
    sa_rewards = np.zeros(steps)
    cs_rewards = np.zeros(steps)
    sa_optimals = np.zeros(steps)
    cs_optimals = np.zeros(steps)

    for _ in range(runs):
        bandit = Bandit(num_arms=10)
        sample_average_agent = Agent(num_arms=10, alpha=None)
        constant_stepsize_agent = Agent(num_arms=10, alpha=0.1)

        for t in range(steps):
            sa_action = sample_average_agent.get_action()
            sa_reward = bandit.pull_arm(sa_action)
            sample_average_agent.update(sa_action, sa_reward)

            cs_action = constant_stepsize_agent.get_action()
            cs_reward = bandit.pull_arm(cs_action)
            constant_stepsize_agent.update(cs_action, cs_reward)

            sa_rewards[t] += sa_reward
            cs_rewards[t] += cs_reward
            sa_optimals[t] += (sa_action == bandit.optimal())
            cs_optimals[t] += (cs_action == bandit.optimal())

            bandit.shuffle()

    return sa_rewards / runs, cs_rewards / runs, 100 * sa_optimals / runs, 100 * cs_optimals / runs


np.random.seed(42)

sa_rewards, cs_rewards, sa_optimals, cs_optimals = run()


def smooth(y, alpha=0.05):
    b = [alpha]
    a = [1, alpha - 1]
    return lfilter(b, a, y)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

ax1.plot(sa_rewards, alpha=0.25, linewidth=1, label="Sample-average (raw)")
ax1.plot(cs_rewards, alpha=0.25, linewidth=1, label="α=0.1 (raw)")
ax1.plot(smooth(np.asarray(sa_rewards)), linewidth=2, label="Sample-average (smoothed)")
ax1.plot(smooth(np.asarray(cs_rewards)), linewidth=2, label="α=0.1 (smoothed)")
ax2.set_xlabel("Steps")
ax1.set_ylabel("Average reward")
ax1.set_title("Average Reward")
ax1.grid(alpha=0.3)
ax1.legend(loc="best")

ax2.plot(sa_optimals, alpha=0.25, linewidth=1, label="Sample-average (raw)")
ax2.plot(cs_optimals, alpha=0.25, linewidth=1, label="α=0.1 (raw)")
ax2.plot(smooth(np.asarray(sa_optimals)), linewidth=2, label="Sample-average (smoothed)")
ax2.plot(smooth(np.asarray(cs_optimals)), linewidth=2, label="α=0.1 (smoothed)")
ax2.set_xlabel("Steps")
ax2.set_ylabel("% Optimal action")
ax2.set_title("Optimal Action %")
ax2.grid(alpha=0.3)
ax2.legend(loc="best")

plt.tight_layout()
plt.show()
