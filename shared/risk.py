"""Business logic: turn model output + news sentiment into a sized trade decision.

Pure functions, no I/O, so the risk service (and tests) can use it directly.
Long-only paper trading: SELL only closes an existing position.
"""
from dataclasses import dataclass

SENTIMENT_SCORE = {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}


@dataclass(frozen=True)
class RiskConfig:
    buy_threshold: float = 0.58        # adjusted p_up needed to BUY
    sell_threshold: float = 0.42       # adjusted p_up at/below which we SELL
    sentiment_weight: float = 0.05     # max shift news can apply to p_up
    max_position_pct: float = 0.10     # max share of equity in one symbol
    max_exposure_pct: float = 0.60     # max share of equity invested overall
    full_size_edge: float = 0.15       # edge (|p-0.5|) at which we use max size
    daily_loss_limit_pct: float = 0.03  # stop opening positions past this loss


@dataclass(frozen=True)
class Portfolio:
    equity: float          # cash + market value of positions
    cash: float
    position_qty: float    # current quantity held in this symbol
    exposure_value: float  # market value of ALL positions
    day_pnl_pct: float     # today's P&L as a fraction of equity (-0.02 = -2%)


@dataclass(frozen=True)
class Decision:
    action: str            # BUY | SELL | HOLD
    quantity: float
    p_adjusted: float
    reason: str


def adjust_probability(p_up: float, sentiment: str, cfg: RiskConfig) -> float:
    """Shift the model's probability by the news sentiment, kept inside [0, 1]."""
    shift = SENTIMENT_SCORE.get(sentiment, 0.0) * cfg.sentiment_weight
    return min(1.0, max(0.0, p_up + shift))


def decide(p_up: float, sentiment: str, price: float,
           pf: Portfolio, cfg: RiskConfig = RiskConfig()) -> Decision:
    p = adjust_probability(p_up, sentiment, cfg)

    if price <= 0 or pf.equity <= 0:
        return Decision("HOLD", 0.0, p, "invalid price or equity")

    # Exiting is always allowed, even when the loss limit is hit.
    if p <= cfg.sell_threshold:
        if pf.position_qty > 0:
            return Decision("SELL", pf.position_qty, p, f"p={p:.2f} <= sell threshold")
        return Decision("HOLD", 0.0, p, "bearish but no position (long-only)")

    if p < cfg.buy_threshold:
        return Decision("HOLD", 0.0, p, f"p={p:.2f} in neutral zone")

    if pf.day_pnl_pct <= -cfg.daily_loss_limit_pct:
        return Decision("HOLD", 0.0, p, "daily loss limit reached")

    # Size: scales with edge, then capped by per-symbol, exposure and cash limits.
    edge = p - 0.5
    target_value = pf.equity * cfg.max_position_pct * min(1.0, edge / cfg.full_size_edge)
    held_value = pf.position_qty * price
    room_symbol = pf.equity * cfg.max_position_pct - held_value
    room_exposure = pf.equity * cfg.max_exposure_pct - pf.exposure_value
    value = min(target_value, room_symbol, room_exposure, pf.cash)

    if value < price:  # cannot afford even one unit
        return Decision("HOLD", 0.0, p, "no room left under risk limits")
    return Decision("BUY", value / price, p, f"p={p:.2f}, size capped by limits")
