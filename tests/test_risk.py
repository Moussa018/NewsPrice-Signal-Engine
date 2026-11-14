import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "shared"))
from risk import Portfolio, RiskConfig, decide  # noqa: E402

CFG = RiskConfig()


def pf(**kw):
    base = dict(equity=10_000, cash=10_000, position_qty=0, exposure_value=0, day_pnl_pct=0)
    return Portfolio(**{**base, **kw})


def test_neutral_zone_holds():
    assert decide(0.50, "neutral", 100, pf()).action == "HOLD"


def test_strong_signal_buys_within_cap():
    d = decide(0.70, "neutral", 100, pf())
    assert d.action == "BUY"
    assert d.quantity * 100 <= 10_000 * CFG.max_position_pct + 1e-9


def test_bearish_news_can_cancel_a_buy():
    assert decide(0.60, "bearish", 100, pf()).action == "HOLD"


def test_sell_closes_position():
    d = decide(0.30, "neutral", 100, pf(position_qty=5))
    assert (d.action, d.quantity) == ("SELL", 5)


def test_bearish_without_position_holds():
    assert decide(0.30, "neutral", 100, pf()).action == "HOLD"


def test_daily_loss_blocks_buy_but_not_sell():
    assert decide(0.70, "neutral", 100, pf(day_pnl_pct=-0.05)).action == "HOLD"
    assert decide(0.30, "neutral", 100, pf(day_pnl_pct=-0.05, position_qty=3)).action == "SELL"


def test_exposure_cap():
    d = decide(0.70, "neutral", 100, pf(exposure_value=6_000))
    assert d.action == "HOLD"
