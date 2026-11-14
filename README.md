# News Price Signal Engine
A trading signal pipeline built from small services. It reads prices and news from Kafka and outputs a BUY, SELL or HOLD signal with a position size. Paper trading only.

## How it works

```
News ──► Laya service ──────────────┐
                                    ├──► Risk service ──► Trade log ◄── Gateway ◄── Client
Prices ► Features ► XGBoost model ──┘
```

| Service | Job |
|---|---|
| Market data | Fetches prices and publishes them to Kafka |
| Feature service | Computes RSI, MACD, moving averages, volatility |
| Inference | XGBoost predicts the chance that price goes up |
| Laya service | Labels news headlines as bullish, bearish or neutral |
| Risk service | Combines both inputs, applies limits, sets position size |
| Trade log | Stores every signal and paper-trading results |
| Gateway + Auth | API keys and `GET /signals` |

Monitoring uses Prometheus and Grafana.

## Decision logic

Two layers sit in the risk service:

- **Risk rules** (`shared/risk.py`): combines the model's probability of a rise with the news sentiment, then applies limits (max position per symbol, max total exposure, cash, daily loss stop) to output BUY, SELL or HOLD with a size. Long-only.
- **Markov Decision Process** (`shared/mdp.py`): trading modelled as an MDP.
  - State: probability bucket, sentiment, in position or not.
  - Actions: HOLD, BUY, SELL (masked to stay long-only).
  - Reward: position return, minus fees, minus a variance penalty for risk aversion.
  - A tabular Q-learning agent learns the policy. It only proposes an action; the risk rules always have the final say.

Sentiment history is not available for training, so the MDP is trained with neutral sentiment for now.

## Stack

Python, FastAPI, Kafka, XGBoost, Laya, SQLite, Docker Compose.

## Status

Planning. Build order:

1. Shared features and model training
2. Kafka and the price path
3. Risk service and trade log
4. Laya news path
5. Gateway, auth and monitoring

## Run

Not available yet. Once built: `docker compose up`.
