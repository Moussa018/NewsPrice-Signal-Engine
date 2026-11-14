"""Trading as a Markov Decision Process (long-only, one symbol).

State  : (p_bin, sentiment, in_position)   p_bin in 0..N_BINS-1, sentiment in {-1,0,1}
Actions: HOLD=0, BUY=1, SELL=2   (BUY only when flat, SELL only when long)
Reward : pos * ret - fee * |trade| - risk_aversion * (pos * ret)^2
The learned policy only *proposes* an action; risk.decide() still enforces limits.
"""
from dataclasses import dataclass

import numpy as np

HOLD, BUY, SELL = 0, 1, 2
ACTION_NAMES = ["HOLD", "BUY", "SELL"]
N_BINS = 5
N_STATES = N_BINS * 3 * 2


def p_to_bin(p_up: float) -> int:
    return min(N_BINS - 1, max(0, int(p_up * N_BINS)))


def encode(p_up: float, sentiment: int, in_position: int) -> int:
    return (p_to_bin(p_up) * 3 + (sentiment + 1)) * 2 + in_position


def valid_actions(in_position: int) -> list[int]:
    return [HOLD, SELL] if in_position else [HOLD, BUY]


@dataclass
class TradingEnv:
    p_up: np.ndarray        # model probability per step
    returns: np.ndarray     # next-step return realised after acting at step t
    sentiment: np.ndarray | None = None
    fee: float = 0.001
    risk_aversion: float = 5.0

    def reset(self) -> int:
        self.t, self.pos = 0, 0
        return self._state()

    def _state(self) -> int:
        s = 0 if self.sentiment is None else int(self.sentiment[self.t])
        return encode(float(self.p_up[self.t]), s, self.pos)

    def step(self, action: int):
        if action not in valid_actions(self.pos):
            raise ValueError(f"illegal action {ACTION_NAMES[action]} (pos={self.pos})")
        trade = 1 if action in (BUY, SELL) else 0
        if action == BUY:
            self.pos = 1
        elif action == SELL:
            self.pos = 0
        # Position held while the next return is realised (SELL exits before it).
        pnl = self.pos * float(self.returns[self.t])
        reward = pnl - self.fee * trade - self.risk_aversion * pnl ** 2
        self.t += 1
        done = self.t >= len(self.returns)
        return (None if done else self._state()), reward, done


def train_q(env: TradingEnv, episodes: int = 300, alpha: float = 0.1, gamma: float = 0.95,
            eps_start: float = 1.0, eps_end: float = 0.05, seed: int = 0) -> np.ndarray:
    """Tabular Q-learning with epsilon-greedy exploration. Returns Q[N_STATES, 3]."""
    rng = np.random.default_rng(seed)
    q = np.zeros((N_STATES, 3))
    for ep in range(episodes):
        eps = eps_start + (eps_end - eps_start) * ep / max(1, episodes - 1)
        s = env.reset()
        done = False
        while not done:
            acts = valid_actions(env.pos)
            a = int(rng.choice(acts)) if rng.random() < eps else max(acts, key=lambda x: q[s, x])
            s2, r, done = env.step(a)
            target = r if done else r + gamma * max(q[s2, x] for x in valid_actions(env.pos))
            q[s, a] += alpha * (target - q[s, a])
            s = s2 if s2 is not None else s
    return q


def policy(q: np.ndarray, p_up: float, sentiment: int, in_position: int) -> int:
    s = encode(p_up, sentiment, in_position)
    return max(valid_actions(in_position), key=lambda a: q[s, a])
