# 🧠 Roostoo Crypto Trading Bot

This repository contains the **full implementation of a live-deployed crypto trading bot**, developed and used during a competitive trading hackathon. The bot employs real-time algorithmic decision-making using a technically sound, fully auditable strategy based on **Moving Averages**, **RSI**, **ADX**, and **systematic risk controls**.

The bot secured **Second place** in the competition by generating the **maximum realized profit** among all participants. This repo includes the **exact code** used during the event.

During the competition, we observed our bot's performance and to enhace the bot's performance and to make sure that it was running efficiently, we adjusted some parameters to make the bot more efficient.

---

## 🚀 Core Strategy

### 📌 Entry Conditions
- **Moving Average Crossover**
  - Buy when **5-period MA > 15-period MA**
  - Confirm trend momentum with RSI and ADX

- **RSI (Relative Strength Index)**
  - Buy if **RSI > 47** (indicates positive momentum)

- **ADX (Average Directional Index)**
  - Trade only when **ADX > 15** (ensures strong trend presence)

### 📌 Exit Conditions
- **Profit Target**: Automatically sell if **net profit exceeds +1.2%** (adjusted for commission)
- **Stop Loss**: Exit if **net loss reaches -0.6%** (protects against trend reversals)
- **Trailing Stop**:
  - Activates when profit reaches **+0.8%**
  - Trails the price by **0.5%** to lock in gains
- **Max Holding Period**: Force-sell any position held for **more than 12 hours**

---

## 🧠 Bot Architecture

### Key Components
- `TradingBot`: Orchestrates signal generation, portfolio management, order placement
- `MovingAverageCrossoverStrategy`: Core signal engine (MA, RSI, ADX)
- `RoostooAPIClient`: Handles mock trading API (compatible with Roostoo platform)
- `RiskManager`: Evaluates P&L, Sharpe Ratio, and other metrics

### Workflow
1. Fetches real-time prices from API
2. Calculates technical indicators
3. Generates buy/sell/hold signal
4. Applies exit logic on open positions
5. Executes trade via mock API
6. Logs every trade and decision reason

---

## 📁 File Structure
```Roostoo_Trading_Bot_Mavericks
├── trading_bot.py            # Main execution script
├── README.md                 # Project documentation
```

---

## 📈 Performance Metrics
- Real-time equity tracking
- Sharpe Ratio based on daily returns
- Complete trade history logging

Each log entry includes:
- Timestamp
- Symbol
- Entry/exit price
- Quantity
- Commission
- Reason for trade

---

## 🏆 Achievements
- 🥇 **Second Place** in crypto hackathon
- ✅ 100% trade logic alignment with declared indicators
- 

---

## 👨‍💻 Authors & Credits
Built by a passionate team Mavericks from **Rajiv Gandhi Institute of Petroleum Technology (RGIPT)** as part of a IEEE RGIPT KodeKurrent hackathon.
## Mavericks Team Members - Apurv Gupta (Team Leader), Kaushal Sharma, Abhigyan Pandey and Sumit Vatsa


---
