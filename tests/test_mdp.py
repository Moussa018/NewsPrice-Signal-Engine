import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from mdp import BUY, HOLD, SELL, TradingEnv, policy, train_q  # noqa: E402


def make_env(n=2000, seed=1):
    """Synthetic world: high p_up really is followed by a positive return."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 1, n)
    ret = (p - 0.5) * 0.02 + rng.normal(0, 0.002, n)
    return TradingEnv(p_up=p, returns=ret)


def test_illegal_actions_rejected():
    env = make_env()
    env.reset()
    try:
        env.step(SELL)  # flat -> cannot sell
    except ValueError:
        return
    raise AssertionError("SELL while flat should raise")


def test_learns_to_buy_high_p_and_stay_out_on_low_p():
    q = train_q(make_env(), episodes=200)
    assert policy(q, 0.95, 0, 0) == BUY
    assert policy(q, 0.05, 0, 0) == HOLD
    assert policy(q, 0.05, 0, 1) == SELL
