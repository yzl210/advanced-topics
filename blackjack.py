from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

random = np.random


def draw_card():
    return random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])


def usable_ace(hand):
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    total = sum(hand)
    return total + 10 if usable_ace(hand) else total


def is_bust(hand):
    return sum_hand(hand) > 21


def score(hand):
    return 0 if is_bust(hand) else sum_hand(hand)


def dealer_policy(dealer_hand):
    return STICK if sum_hand(dealer_hand) >= 17 else HIT


HIT = 0
STICK = 1
ACTIONS = [HIT, STICK]

PLAYER_SUMS = list(range(12, 22))
DEALER_SHOWING = list(range(1, 11))
USE_ACE = [True, False]


def random_start():
    player_sums = random.choice(PLAYER_SUMS)
    dealer_show = random.choice(DEALER_SHOWING)
    use_ace = random.choice(USE_ACE)
    action = random.choice(ACTIONS)

    hand = []
    if use_ace:
        x = player_sums - 11
        x = min(max(1, x), 10)
        hand = [1, x]
    else:
        for _ in range(20):
            c1 = random.randint(2, 10)
            c2 = player_sums - c1
            if 1 <= c2 <= 10:
                if not (c1 == 1 or c2 == 1):
                    hand = [c1, c2]
                    break
        if not hand:
            remain = player_sums
            while remain > 0:
                c = min(10, max(2, remain - 2)) if remain > 3 else remain
                if c == 1: c = 2
                hand.append(c)
                remain -= c
    dealer_hand = [dealer_show, draw_card()]
    return (player_sums, dealer_show, use_ace), action, hand, dealer_hand


def get_state(player_hand, dealer_show):
    ps = sum_hand(player_hand)
    ua = usable_ace(player_hand)
    return ps, dealer_show, ua


def generate_episode_with_policy(Q):
    (player_sum, dealer_show, ua), a0, player_hand, dealer_hand = random_start()
    state = get_state(player_hand, dealer_show)
    a = a0
    episode = []

    while True:
        episode.append((state, a))
        if a == STICK:
            break

        player_hand.append(draw_card())
        player_sum = sum_hand(player_hand)

        if player_sum == 21:
            reward = 1.5
            return episode, reward

        if is_bust(player_hand):
            reward = -1
            return episode, reward

        state = get_state(player_hand, dealer_show)
        hit_q = Q[(state, HIT)]
        stick_q = Q[(state, STICK)]
        a = HIT if hit_q >= stick_q else STICK

    while dealer_policy(dealer_hand) == HIT:
        dealer_hand.append(draw_card())

    if is_bust(dealer_hand):
        reward = 1
    else:
        player_score = score(player_hand)
        dealer_score = score(dealer_hand)

        if player_score == dealer_score:
            reward = -1
        else:
            reward = 1 if player_score > dealer_score else -1

    return episode, reward


def run(episodes=200000):
    Q = defaultdict(float)
    returns = defaultdict(list)
    for ep in range(episodes):
        episode, G = generate_episode_with_policy(Q)
        visited = set()
        for i, (s, a) in enumerate(episode):
            if (s, a) in visited:
                continue
            visited.add((s, a))
            returns[(s, a)].append(G)
            Q[(s, a)] = np.mean(returns[(s, a)])
    return Q


Q = run(episodes=1000000)


def derive_policy_and_value(Q):
    policy_ua = np.zeros((10, 10), dtype=int)
    policy_no = np.zeros((10, 10), dtype=int)
    value_ua = np.zeros((10, 10))
    value_no = np.zeros((10, 10))

    for i, ps in enumerate(range(12, 22)):
        for j, ds in enumerate(range(1, 11)):
            s_ua = (ps, ds, True)
            s_no = (ps, ds, False)
            hit_q = Q[(s_ua, HIT)]
            stick_q = Q[(s_ua, STICK)]
            policy_ua[i, j] = STICK if stick_q > hit_q else HIT
            value_ua[i, j] = max(hit_q, stick_q)

            hit_q = Q[(s_no, HIT)]
            stick_q = Q[(s_no, STICK)]
            policy_no[i, j] = STICK if stick_q > hit_q else HIT
            value_no[i, j] = max(hit_q, stick_q)

    return policy_ua, policy_no, value_ua, value_no


policy_ua, policy_no, value_ua, value_no = derive_policy_and_value(Q)

fig = plt.figure(figsize=(12, 10))

# Policy (usable ace)
ax1 = fig.add_subplot(221)
im1 = ax1.imshow(policy_ua, extent=[1, 10, 12, 21], aspect='auto', origin='lower')
ax1.set_xlabel('Dealer showing')
ax1.set_ylabel('Player sum')
ax1.set_title('Optimal policy (Usable ace)')
fig.colorbar(im1, ax=ax1, label='Action (0=HIT, 1=STICK)')

# Policy (no usable ace)
ax2 = fig.add_subplot(222)
im2 = ax2.imshow(policy_no, extent=[1, 10, 12, 21], aspect='auto', origin='lower')
ax2.set_xlabel('Dealer showing')
ax2.set_ylabel('Player sum')
ax2.set_title('Optimal policy (No usable ace)')
fig.colorbar(im2, ax=ax2, label='Action (0=HIT, 1=STICK)')

# Value surface (usable ace)
ax3 = fig.add_subplot(223, projection='3d')
X, Y = np.meshgrid(np.arange(1, 11), np.arange(12, 22))
Z = value_ua
ax3.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax3.set_xlabel('Dealer showing')
ax3.set_ylabel('Player sum')
ax3.set_zlabel('Value')
ax3.set_title('State-value (Usable ace)')

# Value surface (no usable ace)
ax4 = fig.add_subplot(224, projection='3d')
Z = value_no
ax4.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
ax4.set_xlabel('Dealer showing')
ax4.set_ylabel('Player sum')
ax4.set_zlabel('Value')
ax4.set_title('State-value (No usable ace)')

plt.tight_layout()
plt.show()
