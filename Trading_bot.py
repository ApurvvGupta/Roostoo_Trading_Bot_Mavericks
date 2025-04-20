import time
import hmac
import hashlib
import requests
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import sys
import pytz
import json
import os

class Formatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, timezone=None):
        super().__init__(fmt, datefmt)
        self.timezone = timezone

    def converter(self, timestamp):     
        dt = datetime.fromtimestamp(timestamp, tz=self.timezone)
        return dt

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()

TIMEZONE = pytz.timezone("Asia/Kolkata")
formatter = Formatter(
    fmt='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    timezone=TIMEZONE
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
    force=True
)

API_BASE_URL = "https://mock-api.roostoo.com/v3"
API_KEY = "sX8cRXeqEjaOktC4MX2R8ovMg7Hkq8PzEw2CgBTKu9lHScVryD4eaA0z7u5BQinN"
SECRET_KEY = "trocsegSyuxxkVnOE4sUxSndp09inv2DgqMvrrN7YbEiTyRpbg4qlLQoRWSSuJ0f"
TRADING_INTERVAL = 45
INITIAL_BALANCE = 50000
RUN_DURATION = 43200
COMMISSION_RATE = 0.001

CRYPTOS = [
    {"symbol": "ETH/USD", "name": "Ethereum", "weight": 0.25},
    {"symbol": "BTC/USD", "name": "Bitcoin", "weight": 0.3},
    {"symbol": "AVAX/USD", "name": "Avalanche", "weight": 0.02},
    {"symbol": "LINK/USD", "name": "Chainlink", "weight": 0.1},
    {"symbol": "GALA/USD", "name": "GALA", "weight": 0.02},
    {"symbol": "USTC/USD", "name": "TerraClassicUSD", "weight": 0.2},
    {"symbol": "SOL/USD", "name": "SOLANA", "weight": 0.02},
    {"symbol": "EPIC/USD", "name": "Epic Chain", "weight": 0.05}
    
]

COMMISSION_RATE = 0.001  
PROFIT_THRESHOLD = 0.012 + 2 * COMMISSION_RATE      
STOP_LOSS_THRESHOLD = -0.006 + 2 * COMMISSION_RATE  
TRAILING_STOP_ACTIVATION = 0.008                    
TRAILING_STOP_PERCENTAGE = 0.005                    
MAX_HOLDING_PERIOD = 3600 * 12                      

STEP_SIZES = {
    "ETH/USD": 0.01,
    "BTC/USD": 0.001,
    "AVAX/USD": 0.1,
    "LINK/USD": 0.1,
    "GALA/USD": 1,
    "USTC/USD": 1,
    "SOL/USD": 0.01,
    "EPIC/USD": 1,
}

TRADE_DATA_FILE = "trade_data.json"


def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50 
    deltas = np.diff(prices[-(period+1):])
    gains = deltas[deltas > 0].sum() / period
    losses = -deltas[deltas < 0].sum() / period
    rs = gains / losses if losses != 0 else 0
    return 100 - (100 / (1 + rs))

def calculate_adx(highs, lows, closes, period=14):
    if len(closes) < period * 2:
        return 0

    tr_list = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

        up_move = high - prev_high
        down_move = prev_low - low
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

    tr14 = np.zeros(len(tr_list))
    plus_dm14 = np.zeros(len(plus_dm))
    minus_dm14 = np.zeros(len(minus_dm))

    tr14[period - 1] = sum(tr_list[:period])
    plus_dm14[period - 1] = sum(plus_dm[:period])
    minus_dm14[period - 1] = sum(minus_dm[:period])

    for i in range(period, len(tr_list)):
        tr14[i] = tr14[i - 1] - (tr14[i - 1] / period) + tr_list[i]
        plus_dm14[i] = plus_dm14[i - 1] - (plus_dm14[i - 1] / period) + plus_dm[i]
        minus_dm14[i] = minus_dm14[i - 1] - (minus_dm14[i - 1] / period) + minus_dm[i]

    plus_di14 = 100 * (plus_dm14 / tr14)
    minus_di14 = 100 * (minus_dm14 / tr14)
    dx = 100 * (np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14))

    adx = np.zeros(len(dx))
    adx[period * 2 - 1] = np.mean(dx[period - 1 : period * 2 - 1])
    for i in range(period * 2, len(dx)):
        adx[i] = ((adx[i - 1] * (period - 1)) + dx[i]) / period
    return adx[-1]


class RoostooAPIClient:
    def __init__(self, api_key, secret_key, base_url=API_BASE_URL):
        self.api_key = api_key
        self.secret_key = secret_key.encode()
        self.base_url = base_url

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _sign(self, params: dict):
        sorted_items = sorted(params.items())
        query_string = '&'.join([f"{key}={value}" for key, value in sorted_items])
        signature = hmac.new(self.secret_key, query_string.encode(), hashlib.sha256).hexdigest()
        return signature, query_string

    def _headers(self, params: dict, is_signed=False):
        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }
        if is_signed:
            signature, _ = self._sign(params)
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = signature
        return headers

    def _handle_response(self, response):
        if response.status_code != 200:
            logging.error(f"HTTP Error: {response.status_code} {response.text}")
            logging.error(f"Request URL: {response.url}")
            return None
        try:
            data = response.json()
        except Exception as e:
            logging.error(f"JSON decode error: {e}")
            return None
        return data

    def get_ticker(self, pair=None):
        url = f"{self.base_url}/ticker"
        params = {
            "timestamp": self._get_timestamp()
        }
        if pair:
            params["pair"] = pair
        headers = self._headers(params, is_signed=False)
        response = requests.get(url, params=params, headers=headers)
        return self._handle_response(response)

    def place_order(self, symbol, quantity, order_type):
        url = f"{self.base_url}/place_order"
        params = {
            "pair": symbol,
            "quantity": quantity,
            "side": "BUY" if order_type == "buy" else "SELL",
            "type": "MARKET",
            "timestamp": self._get_timestamp()
        }
        headers = self._headers(params, is_signed=True)
        response = requests.post(url, data=params, headers=headers)
        logging.info(f"API Response: {response.text}")
        return self._handle_response(response)

    def get_balance(self):
        """Fetch current balance from the exchange."""
        url = f"{self.base_url}/balance"
        params = {
            "timestamp": self._get_timestamp()
        }
        headers = self._headers(params, is_signed=True)
        response = requests.get(url, params=params, headers=headers)
        return self._handle_response(response)

class MovingAverageCrossoverStrategy:
    def __init__(self, short_window=5, long_window=15, rsi_period=14, adx_period=7):
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_period = rsi_period
        self.adx_period = adx_period
        self.prices = []
        self.highs = []
        self.lows = []

    def update_price(self, price, high=None, low=None):
        self.prices.append(price)
        if high is not None: self.highs.append(high)
        if low is not None: self.lows.append(low)

        max_len = max(self.long_window, self.rsi_period, self.adx_period)
        if len(self.prices) > max_len: self.prices.pop(0)
        if len(self.highs) > max_len: self.highs.pop(0)
        if len(self.lows) > max_len: self.lows.pop(0)

    def generate_signal(self):
    if len(self.prices) < max(self.long_window, self.rsi_period, self.adx_period):
        return "HOLD"
    short_ma = np.mean(self.prices[-self.short_window:])
    long_ma = np.mean(self.prices[-self.long_window:])
    rsi = calculate_rsi(self.prices, self.rsi_period)
    adx = calculate_adx(self.highs, self.lows, self.prices, self.adx_period)

    logging.info(f"Short MA: {short_ma:.2f}, Long MA: {long_ma:.2f}, RSI: {rsi:.2f}, ADX: {adx:.2f}")

    if short_ma > long_ma and rsi > 47 and adx > 15:
        return "BUY"
    elif short_ma < long_ma and rsi < 53 and adx > 15:
        return "SELL"
    else:
        return "HOLD"

class RiskManager:
    def __init__(self):
        self.portfolio_values = []

    def update_portfolio(self, value):
        self.portfolio_values.append(value)

    def calculate_sharpe_ratio(self):
        if len(self.portfolio_values) < 2:
            return 0
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        excess_returns = returns - RISK_FREE_RATE
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns)
        if std_return == 0:
            return 0
        sharpe_ratio = mean_return / std_return
        return sharpe_ratio

class TradingBot:
    def __init__(self, api_client, initial_cash=INITIAL_BALANCE):
        self.api_client = api_client
        self.cash = initial_cash
        self.holdings = {crypto["symbol"]: 0.0 for crypto in CRYPTOS}
        self.trade_log = []
        self.strategies = {crypto["symbol"]: MovingAverageCrossoverStrategy() for crypto in CRYPTOS}
        self.risk_manager = RiskManager()
        self.entry_prices = {crypto["symbol"]: 0.0 for crypto in CRYPTOS}
        self.historical_prices = {crypto["symbol"]: [] for crypto in CRYPTOS} 
        self.trailing_stops = {crypto["symbol"]: None for crypto in CRYPTOS}  
        self.sync_holdings() 
        self.load_trade_data()

    def sync_holdings(self):
        """Sync the bot's holdings with the actual balance on the exchange."""
        balance_data = self.api_client.get_balance()
        logging.info(f"Balance API Response: {json.dumps(balance_data, indent=2)}")
        if balance_data and balance_data.get("Success"):
            if "SpotWallet" in balance_data:
                for crypto in CRYPTOS:
                    symbol = crypto["symbol"]
                    coin = symbol.split("/")[0]
                    if coin in balance_data["SpotWallet"]:
                        exchange_balance = float(balance_data["SpotWallet"][coin]["Free"])
                        if exchange_balance < 0:
                            logging.warning(f"Negative holdings detected for {symbol}. Resetting to 0.")
                            exchange_balance = 0.0
                        if self.holdings[symbol] != exchange_balance:
                            logging.warning(f"Discrepancy detected for {symbol}. Bot's holdings: {self.holdings[symbol]:.4f}, Exchange balance: {exchange_balance:.4f}")
                            self.holdings[symbol] = exchange_balance
                        logging.info(f"Synced holdings for {symbol}: {self.holdings[symbol]:.4f}")
                    else:
                        logging.warning(f"Coin {coin} not found in SpotWallet. Setting holdings to 0.")
                        self.holdings[symbol] = 0.0
            else:
                logging.error("'SpotWallet' key not found in balance API response.")
        else:
            logging.error("Failed to fetch balance from the exchange.")

    def load_trade_data(self):
        if os.path.exists(TRADE_DATA_FILE):
            with open(TRADE_DATA_FILE, "r") as file:
                data = json.load(file)
                self.cash = data.get("cash", INITIAL_BALANCE)
                self.holdings = data.get("holdings", {crypto["symbol"]: 0.0 for crypto in CRYPTOS})
                self.trade_log = data.get("trade_log", [])
                self.entry_prices = data.get("entry_prices", {crypto["symbol"]: 0.0 for crypto in CRYPTOS})
                self.latest_prices = data.get("latest_prices", {})
                self.historical_prices = data.get("historical_prices", {crypto["symbol"]: [] for crypto in CRYPTOS})
                self.trailing_stops = {crypto["symbol"]: None for crypto in CRYPTOS} 
                logging.info("Loaded previous trade data.")

    def save_trade_data(self):
        data = {
            "cash": self.cash,
            "holdings": self.holdings,
            "trade_log": self.trade_log,
            "entry_prices": self.entry_prices,
            "latest_prices": getattr(self, "latest_prices", {}),
            "historical_prices": self.historical_prices
        }
        with open(TRADE_DATA_FILE, "w") as file:
            json.dump(data, file)
        logging.info("Saved trade data.")

    def update_portfolio_value(self, prices):
        portfolio_value = self.cash
        for crypto in CRYPTOS:
            portfolio_value += self.holdings[crypto["symbol"]] * prices[crypto["symbol"]]
        self.risk_manager.update_portfolio(portfolio_value)
        return portfolio_value

    def execute_trade(self, symbol, signal, price):
        current_holdings = self.holdings[symbol]
        entry_price = self.entry_prices[symbol]

        if current_holdings > 0:
            exit_condition = self._check_exit_conditions(symbol, price, entry_price)
            if exit_condition:
                self._execute_smart_exit(symbol, price, exit_condition)
                return  

        trade_amount = (self.cash * CRYPTOS[next(i for i, c in enumerate(CRYPTOS) if c["symbol"] == symbol)]["weight"]) / price
        step_size = STEP_SIZES.get(symbol, 0.01)
        trade_amount = (trade_amount // step_size) * step_size
        min_order_size = step_size
        if trade_amount < min_order_size:
            trade_amount = min_order_size

        logging.info(f"Trade amount for {symbol}: {trade_amount:.4f} | Current holdings: {self.holdings[symbol]:.4f}")

        if signal == "BUY" and self.cash >= trade_amount * price:
            if self.historical_prices[symbol]:
                historical_avg = np.mean(self.historical_prices[symbol])
                if price < historical_avg:
                    order = self.api_client.place_order(symbol, trade_amount, 'buy')
                    if order and order.get("Success"):
                        self.holdings[symbol] += trade_amount
                        self.cash -= trade_amount * price * (1 + COMMISSION_RATE)
                        self.entry_prices[symbol] = price
                        self.trade_log.append({"timestamp": datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'), "action": "BUY", "symbol": symbol, "price": price, "amount": trade_amount})
                        logging.info(f"Executed BUY: {trade_amount} {symbol} at {price}")
                    else:
                        logging.error(f"Failed to execute BUY order for {symbol}. Error: {order.get('ErrMsg', 'Unknown error')}")
                else:
                    logging.info(f"No BUY executed for {symbol} (current price {price} is not lower than historical average {historical_avg:.2f}).")
            else:
                order = self.api_client.place_order(symbol, trade_amount, 'buy')
                if order and order.get("Success"):
                    self.holdings[symbol] += trade_amount
                    self.cash -= trade_amount * price * (1 + COMMISSION_RATE)
                    self.entry_prices[symbol] = price
                    self.trade_log.append({"timestamp": datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S'), "action": "BUY", "symbol": symbol, "price": price, "amount": trade_amount})
                    logging.info(f"Executed BUY: {trade_amount} {symbol} at {price}")
                else:
                    logging.error(f"Failed to execute BUY order for {symbol}. Error: {order.get('ErrMsg', 'Unknown error')}")

    def _check_exit_conditions(self, symbol, current_price, entry_price):
        """Return exit reason if any condition met"""
        raw_pl = (current_price - entry_price) / entry_price if entry_price > 0 else 0
        adjusted_pl = self._calculate_adjusted_pl(entry_price, current_price) if entry_price > 0 else 0

        if adjusted_pl <= STOP_LOSS_THRESHOLD:
            return f"Stop loss triggered ({adjusted_pl:.2%} ≤ {STOP_LOSS_THRESHOLD:.2%})"
        if adjusted_pl >= PROFIT_THRESHOLD:
            return f"Profit target reached ({adjusted_pl:.2%} ≥ {PROFIT_THRESHOLD:.2%})"

        if raw_pl >= TRAILING_STOP_ACTIVATION:
            if not self.trailing_stops[symbol]:
                self.trailing_stops[symbol] = current_price * (1 - TRAILING_STOP_PERCENTAGE)
                logging.info(f"Activated trailing stop for {symbol} at {self.trailing_stops[symbol]:.2f}")
            else:
                new_stop = current_price * (1 - TRAILING_STOP_PERCENTAGE)
                self.trailing_stops[symbol] = max(self.trailing_stops[symbol], new_stop)
            if current_price < self.trailing_stops[symbol]:
                return f"Trailing stop triggered at {self.trailing_stops[symbol]:.2f}"

        if self._position_age(symbol) > MAX_HOLDING_PERIOD:
            return f"Max holding period exceeded ({MAX_HOLDING_PERIOD/3600:.1f}h)"

        if self._needs_rebalancing(symbol, current_price):
            return "Portfolio rebalancing required"

        return None

    def _execute_smart_exit(self, symbol, price, reason):
        """Execute full exit for specific coin with detailed reporting"""
        try:
            step_size = STEP_SIZES[symbol]
            sell_amount = (self.holdings[symbol] // step_size) * step_size
            if sell_amount <= 0:
                logging.error(f"Cannot sell {symbol} - zero position after step adjustment")
                return
            order = self.api_client.place_order(symbol, sell_amount, 'sell')
            if order and order.get("Success"):
                sale_value = sell_amount * price
                commission = sale_value * COMMISSION_RATE
                self.cash += sale_value - commission
                self.holdings[symbol] -= sell_amount
                self._log_trade(symbol, price, sell_amount, commission, reason)
                if self.holdings[symbol] == 0:
                    self.entry_prices[symbol] = 0.0
                    self.trailing_stops[symbol] = None
        except Exception as e:
            logging.error(f"Failed to execute smart exit for {symbol}: {str(e)}")

    def _log_trade(self, symbol, price, amount, commission, reason):
        trade_info = {
            "timestamp": datetime.now(TIMEZONE).isoformat(),
            "symbol": symbol,
            "action": "SELL",
            "price": price,
            "amount": amount,
            "commission": commission,
            "remaining_balance": self.cash,
            "exit_reason": reason
        }
        self.trade_log.append(trade_info)
        logging.info(
            f"Sold {amount:.4f} {symbol} at {price:.2f} | "
            f"Reason: {reason} | "
            f"Commission: {commission:.2f} | "
            f"New Balance: {self.cash:.2f}"
        )

    def _calculate_adjusted_pl(self, entry_price, exit_price):
        """Calculate commission-adjusted P/L percentage"""
        return ((exit_price * (1 - COMMISSION_RATE)) -
               (entry_price * (1 + COMMISSION_RATE))) / (entry_price * (1 + COMMISSION_RATE))

    def _position_age(self, symbol):
        """Calculate time since position was opened"""
        if self.entry_prices[symbol] == 0:
            return 0
        for trade in reversed(self.trade_log):
            if trade["symbol"] == symbol and trade["action"] == "BUY":
                buy_time = datetime.fromisoformat(trade["timestamp"]) if "T" in trade["timestamp"] else datetime.strptime(trade["timestamp"], '%Y-%m-%d %H:%M:%S')
                return (datetime.now(TIMEZONE) - buy_time).total_seconds()
        return 0

    def _needs_rebalancing(self, symbol, current_price):
        """Check if portfolio needs rebalancing"""
        current_value = self.holdings[symbol] * current_price
        portfolio_value = self.cash + sum(self.holdings[c["symbol"]] * self.latest_prices[c["symbol"]]
        for c in CRYPTOS
        )

        target_allocation = CRYPTOS[next(i for i, c in enumerate(CRYPTOS) if c["symbol"] == symbol)]["weight"]
        if portfolio_value == 0:
            return False
        current_allocation = current_value / portfolio_value
        return abs(current_allocation - target_allocation) > 0.05  # 5% threshold

    def run(self, duration_sec=None):
        logging.info("Starting trading bot...")
        start_time = time.time()
        while True:
            try:
                if duration_sec and time.time() - start_time >= duration_sec:
                    logging.info("Trading bot stopped after the specified duration.")
                    break
                prices = {}
                for crypto in CRYPTOS:
                    ticker_data = self.api_client.get_ticker(pair=crypto["symbol"])
                    if ticker_data and ticker_data.get("Success"):
                        price = float(ticker_data["Data"][crypto["symbol"]]["LastPrice"])
                        high = float(ticker_data["Data"][crypto["symbol"]]["High"])
                        low = float(ticker_data["Data"][crypto["symbol"]]["Low"])
                        self.strategies[crypto["symbol"]].update_price(price, high, low)
                        signal = self.strategies[crypto["symbol"]].generate_signal()
                        current_time = datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')
                        logging.info(f"Time: {current_time} | {crypto['name']} | Price: {price} | Signal: {signal}")
                        logging.info(f"Current holdings for {crypto['symbol']}: {self.holdings[crypto['symbol']]:.4f}")
                        logging.info(f"Current cash: {self.cash:.2f}")
                        self.latest_prices = prices  
                        self.historical_prices[crypto["symbol"]].append(price)
                        if len(self.historical_prices[crypto["symbol"]]) > 100:
                            self.historical_prices[crypto["symbol"]].pop(0)
                        self.execute_trade(crypto["symbol"], signal, price)

                portfolio_value = self.update_portfolio_value(prices)
                sharpe_ratio = self.risk_manager.calculate_sharpe_ratio()
                logging.info(f"Portfolio Value: {portfolio_value:.2f}")
                logging.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")

                time.sleep(TRADING_INTERVAL)
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                time.sleep(TRADING_INTERVAL)

        self.save_trade_data()
        self.print_summary()

    def print_summary(self):
        logging.info("--- TRADING BOT SUMMARY ---")
        logging.info(f"Initial Balance: ${INITIAL_BALANCE:.2f}")
        logging.info(f"Final Cash: ${self.cash:.2f}")
        for crypto in CRYPTOS:
            logging.info(f"Final Holdings {crypto['name']}: {self.holdings[crypto['symbol']]:.4f}")
        final_portfolio_value = self.cash
        for crypto in CRYPTOS:
          price = self.historical_prices[crypto["symbol"]][-1] if self.historical_prices[crypto["symbol"]] else 0
          final_portfolio_value += self.holdings[crypto["symbol"]] * price

        logging.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        logging.info(f"Total Trades Executed: {len(self.trade_log)}")
        logging.info(f"Sharpe Ratio: {self.risk_manager.calculate_sharpe_ratio():.4f}")


def main():
    api_client = RoostooAPIClient(API_KEY, SECRET_KEY)
    trading_bot = TradingBot(api_client, initial_cash=INITIAL_BALANCE)
    trading_bot.run(duration_sec=RUN_DURATION)

if __name__ == "__main__":
    main()
