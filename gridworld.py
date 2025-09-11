import matplotlib.pyplot as plt
import numpy as np

grid_size = 5
gamma = 0.9
# A, A_prime, A_reward = (0, 1), (4, 1), 10
# B, B_prime, B_reward = (0, 3), (2, 3), 5
A, A_prime, A_reward = (2, 2), (4, 1), 10
B, B_prime, B_reward = (4, 3), (2, 3), 5
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def act(x, y, action):
    if (x, y) == A:
        return A_prime, A_reward
    if (x, y) == B:
        return B_prime, B_reward
    dx, dy = actions[action]
    new_x, new_y = x + dx, y + dy
    if new_x < 0 or new_x >= grid_size or new_y < 0 or new_y >= grid_size:
        return (x, y), -1
    return (new_x, new_y), 0


grid = np.zeros((grid_size, grid_size))
for _ in range(10000):
    new_grid = np.zeros_like(grid)
    for x in range(grid_size):
        for y in range(grid_size):
            value = 0
            for action in range(len(actions)):
                (new_x, new_y), r = act(x, y, action)
                value += (1 / len(actions)) * (r + gamma * grid[new_x, new_y])
            new_grid[x, y] = value
    grid = new_grid

print(grid)

specials = {
    "A": A,
    "A'": A_prime,
    "B": B,
    "B'": B_prime
}

plt.figure(figsize=(6, 6))
plt.gca().invert_yaxis()

for x in range(grid_size + 1):
    plt.axhline(x - 0.5, color="black", linewidth=1)
    plt.axvline(x - 0.5, color="black", linewidth=1)

for i in range(grid_size):
    for j in range(grid_size):
        label = f"{grid[i, j]:.2f}"
        for k, (si, sj) in specials.items():
            if (i, j) == (si, sj):
                label = f"{k}\n{grid[i, j]:.2f}"
        plt.text(j, i, label, ha="center", va="center", fontsize=18)

plt.xticks([])
plt.yticks([])
plt.axis("equal")
plt.show()
