import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import datetime


class PaperTrader:
    def __init__(self, initial_capital=10000, transaction_cost=0.001, stop_loss_pct=0.05, take_profit_pct=0.10):
        """
        Initialize the PaperTrader.
        :param initial_capital: Starting balance (default: 10,000).
        :param transaction_cost: Transaction cost as a percentage (default: 0.1% per trade).
        :param stop_loss_pct: Percentage for stop-loss (default: 5%).
        :param take_profit_pct: Percentage for take-profit (default: 10%).
        """
        self.cash = initial_capital
        self.positions = {}
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.trade_log = []
        self.total_return = 0
        self.transaction_cost = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def execute_trade(self, action, symbol, price, amount):
        """
        Executes a trade and updates the portfolio, including stop-loss and take-profit.
        :param action: 'buy' or 'sell'
        :param symbol: Stock ticker or asset name
        :param price: Price per unit of asset
        :param amount: Amount to buy/sell
        """
        # Transaction cost is based on the cost of the trade
        cost = price * amount * (1 + self.transaction_cost)
        
        if action == 'buy':
            if self.cash >= cost:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + amount
                self.trade_log.append({'action': 'buy', 'symbol': symbol, 'price': price, 'amount': amount, 'cash_balance': self.cash})
                self._check_stop_loss(symbol, price)  # Check stop-loss after buying
            else:
                print(f"Not enough cash to buy {amount} of {symbol} at {price}.")
        elif action == 'sell':
            if self.positions.get(symbol, 0) >= amount:
                self.cash += price * amount * (1 - self.transaction_cost)
                self.positions[symbol] -= amount
                self.trade_log.append({'action': 'sell', 'symbol': symbol, 'price': price, 'amount': amount, 'cash_balance': self.cash})
            else:
                print(f"Not enough position to sell {amount} of {symbol}.")
        
        # Update portfolio value
        self.update_portfolio_value(price)

    def _check_stop_loss(self, symbol, price):
        """Check if the position has hit stop-loss or take-profit thresholds."""
        entry_price = self.positions.get(symbol, 0) * price  # Buy price * quantity
        if entry_price:
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct)

            if price <= stop_loss_price:
                self.execute_trade('sell', symbol, price, self.positions[symbol])
                print(f"Stop-loss triggered for {symbol}.")
            elif price >= take_profit_price:
                self.execute_trade('sell', symbol, price, self.positions[symbol])
                print(f"Take-profit triggered for {symbol}.")

    def update_portfolio_value(self, price):
        """
        Update the total value of the portfolio.
        """
        total_value = self.cash
        for symbol, amount in self.positions.items():
            total_value += amount * price  # Assuming last price for valuation
        self.portfolio_value = total_value

    def get_portfolio_value(self):
        return self.portfolio_value

    def get_trade_log(self):
        return self.trade_log


class MarketData:
    def __init__(self, data):
        """
        Initialize the MarketData.
        :param data: Pandas DataFrame with columns 'Date' and 'PRICE'.
        """
        self.data = data
        self.current_index = 0

    def get_next_data_point(self):
        """
        Get the next market data point (price) and the date.
        """
        if self.current_index < len(self.data) - 1:
            data_point = self.data.iloc[self.current_index]
            self.current_index += 1
            return data_point
        else:
            return None

    def get_current_data(self):
        """
        Return the current market data point.
        """
        return self.data.iloc[self.current_index]

    def reset(self):
        """
        Reset to the beginning of the data.
        """
        self.current_index = 0


class Strategy:
    def __init__(self, short_window=40, long_window=100):
        """
        Initialize the Strategy.
        :param short_window: Short-term moving average window (default: 40).
        :param long_window: Long-term moving average window (default: 100).
        """
        self.short_window = short_window
        self.long_window = long_window

    def generate_signal(self, market_data,trader):
        """
        Generate buy or sell signal based on moving average crossover strategy.
        :param market_data: Market data (contains 'PRICE' refering to prices).
        :return: 'buy', 'sell', or 'hold'
        """
        short_ma = market_data['PRICE'].rolling(window=self.short_window).mean().iloc[-1]
        long_ma = market_data['PRICE'].rolling(window=self.long_window).mean().iloc[-1]

        if short_ma > long_ma:
            # trader.execute_trade('buy', 'SBIN', market_data['PRICE'], 10)
            return 'buy'
        elif short_ma < long_ma:
            # trader.execute_trade('sell', 'SBIN', market_data['PRICE'], 10)
            return 'sell'
        else:
            return 'hold'
                


class BacktestEngine:
    def __init__(self, market_data, trader, strategy):
        """
        Initialize the backtest engine.
        :param market_data: The market data object (historical data).
        :param trader: The PaperTrader object.
        :param strategy: The Strategy object.
        """
        self.market_data = market_data
        self.trader = trader
        self.strategy = strategy

    def run(self):
        """
        Run the backtest simulation.
        """
        self.market_data.reset()

        while True:
            data_point = self.market_data.get_next_data_point()
            if data_point is None:
                break

            signal = self.strategy.generate_signal(market_data.data,trader)
            if signal == 'buy':
                self.trader.execute_trade('buy', 'SBIN', data_point['PRICE'], 10)
            elif signal == 'sell':
                self.trader.execute_trade('sell', 'SBIN', data_point['PRICE'], 10)
            else:
                print("hold")
            

        return self.trader.get_trade_log(), self.trader.get_portfolio_value()


# Performance Metrics Functions

def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0):
    """
    Calculate the Sharpe Ratio of the strategy.
    :param portfolio_values: List of portfolio values over time.
    :param risk_free_rate: The risk-free rate, default is 0.
    :return: Sharpe Ratio
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]  # Calculate daily returns
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(portfolio_values):
    """
    Calculate the Maximum Drawdown of the strategy.
    :param portfolio_values: List of portfolio values over time.
    :return: Maximum Drawdown
    """
    peak_value = portfolio_values[0]
    max_drawdown = 0
    for value in portfolio_values:
        drawdown = (peak_value - value) / peak_value
        max_drawdown = max(max_drawdown, drawdown)
        peak_value = max(peak_value, value)
    return max_drawdown

# Get user inputs
def get_user_inputs():
    """
    Prompt the user for inputs related to strategy, risk management, etc.
    :return: A dictionary of user inputs
    """
    print("Enter the parameters for your backtest:")

    initial_capital = float(input("Initial Capital (default 10000): ") or 10000)
    transaction_cost = float(input("Transaction Cost (as percentage, e.g. 0.001 for 0.1%): ") or 0.001)
    stop_loss_pct = float(input("Stop Loss Percentage (e.g. 0.05 for 5%): ") or 0.05)
    take_profit_pct = float(input("Take Profit Percentage (e.g. 0.10 for 10%): ") or 0.10)
    short_window = int(input("Short Window for Moving Average (default 40): ") or 40)
    long_window = int(input("Long Window for Moving Average (default 100): ") or 100)

    # Load historical data
    file_path = str(input("Enter the path to your historical data CSV: ") or "/Users/saichaitanya/Downloads/SBIData.csv")
    data = pd.read_csv(file_path, parse_dates=['DATE'], index_col='DATE')

    return {
        'initial_capital': initial_capital,
        'transaction_cost': transaction_cost,
        'stop_loss_pct': stop_loss_pct,
        'take_profit_pct': take_profit_pct,
        'short_window': short_window,
        'long_window': long_window,
        'data': data
    }

# Main execution
if __name__ == '__main__':
    # Get user inputs
    user_inputs = get_user_inputs()

    # Instantiate objects
    market_data = MarketData(user_inputs['data'])
    trader = PaperTrader(
        initial_capital=user_inputs['initial_capital'],
        transaction_cost=user_inputs['transaction_cost'],
        stop_loss_pct=user_inputs['stop_loss_pct'],
        take_profit_pct=user_inputs['take_profit_pct']
    )
    strategy = Strategy(short_window=user_inputs['short_window'], long_window=user_inputs['long_window'])
    engine = BacktestEngine(market_data, trader, strategy)

    # Run the backtest
    trade_log, final_value = engine.run()

    # Print trade log
    print("Trade Log:")
    for trade in trade_log:
        print(trade)

    # Print final portfolio value
    print(f"\nFinal Portfolio Value: {final_value}")

    # Calculate Performance Metrics
    # Assuming trade_log contains {'action', 'cash_balance', 'symbol', 'price', 'amount'} for each trade.
    portfolio_values = []

    # Iterate through each trade in the log and calculate the portfolio value at each point
    for trade in trade_log:
        # Fetch the price at the time of the trade (using the trade['price'] at the time of buy/sell)
        trade_price = trade['price']

        # Calculate the portfolio value: cash + value of the current positions
        portfolio_value = trade['cash_balance']
        if trade['action'] == 'buy':
            # Add value of the bought position (price * amount)
            portfolio_value += trade_price * trade['amount']
        elif trade['action'] == 'sell':
            # Subtract value of the sold position (price * amount)
            portfolio_value -= trade_price * trade['amount']
        
        portfolio_values.append(portfolio_value)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_values)
    max_drawdown = calculate_max_drawdown(portfolio_values)

    print(f"\nSharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.4f}")

    # Plot Portfolio Value
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Portfolio Value over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()
