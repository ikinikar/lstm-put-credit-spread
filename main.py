# region imports
from AlgorithmImports import *
from QuantConnect.DataSource import *
from algo import LSTM_Algo
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import fsolve
from sklearn.preprocessing import StandardScaler
# endregion

class WellDressedOrangeLemur(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2019, 12, 1)
        self.SetEndDate(2020, 6, 1)
        self.SetCash(100000)
        self.SetBrokerageModel(
            BrokerageName.InteractiveBrokersBrokerage, AccountType.Cash
        )

        self.minimum_premium_pct = self.GetParameter("minimum_premium_pct", 0.05)
        self.K2_to_K1_ratio = self.GetParameter("K2_to_K1_ratio", 0.95)
        self.min_expiry_days = self.GetParameter("min_expiry_days", 50)
        self.stop_loss_pct = self.GetParameter("stop_loss_pct", -2.5)

        self.Schedule.On(
            self.DateRules.EveryDay("SPY"),
            self.TimeRules.Every(timedelta(minutes=120)),
            self.CheckShouldStopLoss,
        )

        # Add SPY equity
        self.spy = self.AddEquity("SPY", Resolution.HOUR).Symbol
        # Add VIX index with daily data
        self.vix = self.AddIndex("VIX", Resolution.HOUR).Symbol

        self.spy_option_symbol = self.AddOption("SPY", Resolution.HOUR).Symbol

        self.Securities[self.spy_option_symbol].SetFilter(-15, 15, self.min_expiry_days, self.min_expiry_days * 2)

        history = self.History(self.spy, 200, Resolution.HOUR)
        
        # Ensure history is in the correct format and retrieve the 'close' column as a pandas Series
        self.close_prices = history['close']
    
        # Set up indicator containers

        # Attach MACD to SPY
        self._macd = self.macd(self.spy, 12, 26, 9, MovingAverageType.Wilders, Resolution.Hour)

        # Attach BollingerBands to SPY
        self.bollinger = self.bb(self.spy, 20, 2, MovingAverageType.Simple, Resolution.Hour)


        # Store historical data for S&P 500 and VIX
        self.sp500_data = []
        self.vix_data = []
        self.indicators_data = []

        # Store your data for preparing LSTM
        self.data = pd.DataFrame()
        self.data_buffer_size = 200
        # Warm-up for accurate calculations
        self.SetWarmUp(timedelta(days=15))  # Warm up for 30 days or as needed

        self.LSTM_Algo = LSTM_Algo()
        self.model = None
        self.latest = None

    def OnData(self, slice):
        # Example: if not invested
            if self.spy in slice.Bars and self.vix in slice.Bars:
                # Get the SPY minute bar
                spy_bar = slice.Bars[self.spy]
                
                # Append relevant data (timestamp, open, high, low, close, volume)
                self.sp500_data.append(
                    spy_bar.Close
                )
                # Ensure that we have enough data before using indicators

                # Get the VIX minute bar
                vix_bar = slice.Bars[self.vix]
                
                # Append relevant data (timestamp, open, high, low, close, volume)
                self.vix_data.append(
                    vix_bar.Close
                )

                # Optionally: Limit the size of the data for memory efficiency
            if len(self.vix_data) > 500:  # Example: Keep only the latest 10,000 entries #MESS AROUND WITH THIS
                self.vix_data.pop(0)
            # Optionally: Limit the size of the data for memory efficiency
            if len(self.sp500_data) > 500:  # Example: Keep only the latest 10,000 entries #MESS AROUND WITH THIS
                self.sp500_data.pop(0)
            if(len(self.indicators_data) > 500):
                self.indicators_data.pop(0)
            
            if len(self.close_prices) > 30 and self.spy in slice.Bars:  # Ensure we have enough historical data
                spy_close = slice[self.spy].Close

                # Use pd.concat to append the new close price
                new_data = pd.Series([spy_close], index=[pd.to_datetime(slice.Time)])
                self.close_prices = pd.concat([self.close_prices, new_data])

                # Attach MACD to SPY
                self._macd = self.macd(self.spy, 12, 26, 9, MovingAverageType.Wilders, Resolution.Hour)

                # Attach BollingerBands to SPY
                self.bollinger = self.bb(self.spy, 20, 2, MovingAverageType.Simple, Resolution.Hour)

    
                macd_value = self._macd.Current.Value
                macd_signal = self._macd.Signal.Current.Value
                macd_histogram = self._macd.Fast.Current.Value - self._macd.Slow.Current.Value
                bollinger_upper = self.bollinger.UpperBand.Current.Value
                bollinger_lower = self.bollinger.LowerBand.Current.Value



                # Append the indicator values to the list
                self.indicators_data.append([
                        macd_value,
                        macd_signal,
                        macd_histogram,
                        bollinger_upper,
                        bollinger_lower
                ])
            

                # When we have enough data (i.e., 200 data points to ensure smooth calculation)
            if len(self.sp500_data) > self.data_buffer_size and (self.Time.hour == 12):
                X, y = self.PrepareData()
                X_train, X_val, X_test, Y_train, Y_val, Y_test = self.TrainTestSplit(X, y)
                model, test_accuracy = self.train_test_LSTM(X_train, X_val, X_test, Y_train, Y_val, Y_test)
                if(test_accuracy >= 0.5):
                    self.model = model
                else:
                    self.model = None #Don't want to use a model that is worse than a coin flip

                if(self.model is not None and self.latest is not None):
                    predictions = self.model.predict(self.latest)

                    if(predictions[0] > 0.5):
                        
                        if option_chain := slice.OptionChains.get(self.spy_option_symbol):
                            if len(self.GetOptionPositions(self.spy_option_symbol)) == 0:
                                #self.Log("No positions are open, rebalancing")
                                self.Rebalance(option_chain)
                    else:
                        self.Liquidate(self.spy)
                else:
                    self.Liquidate(self.spy)




    def Rebalance(self, option_chain: OptionChain):
        k1_contract, k2_contract = self.SearchForStrikePairs(option_chain)

        # do something with the contracts information to see whether it is worth trading it or not

        if k1_contract is None or k2_contract is None:
            return
        self.OpenPutSpread(k1_contract, k2_contract)


    def SearchForStrikePairs(self, option_chain: OptionChain):
        # Return the strikes and premiums for K1 and K2.

        # Sort contracts by their strike price and filter out the calls.
        puts = [i for i in option_chain if i.Right == OptionRight.Put]
        puts = sorted(puts, key=lambda x: x.Strike, reverse=True)

        #self.Log(f"Found {len(puts)} put contracts")

        # Loop over the puts list and find the first pair that satisfies the minimum premium requirement.
        for i in range(len(puts), 1, -1):
            for j in range(i + 1, len(puts)):
                if (
                    puts[j].Strike >= puts[i].Strike
                    or puts[j].Strike
                    <= self.K2_to_K1_ratio
                    * puts[
                        i
                    ].Strike  # Might want to test making this just less than? May be too rare to matter.
                ):
                    continue

                contract_i = puts[i]
                contract_j = puts[j]

                if contract_i.BidPrice > 0 and contract_j.AskPrice > 0:
                    premium_i = contract_i.BidPrice
                    premium_j = contract_j.AskPrice

                    # If the net premium is greater than or equal to the minimum premium, return the pair.
                    net_premium_dollars = (premium_i - premium_j) * 100
                    maximum_risk = (contract_i.Strike - contract_j.Strike) * 100
                    percent_return = net_premium_dollars / maximum_risk
                    if percent_return >= self.minimum_premium_pct:
                        return contract_i, contract_j

        #self.Log("No suitable option pair found, skipping today")
        return None, None


    # Give 2 contracts and opens a spread
    def OpenPutSpread(self, k1_contract: OptionContract, k2_contract: OptionContract):
        self.SetHoldings(k1_contract.Symbol, -1.0)
        self.MarketOrder(
            k2_contract.Symbol, self.Portfolio[k1_contract.Symbol].Quantity
        )
        #self.Log(
        #    f"Opened put spread, short {k1_contract.Symbol}, long {k2_contract.Symbol}"
        #)

    def CheckShouldStopLoss(self):
        # Check if the market is open for any of the equities
        market_open = self.IsMarketOpen(self.spy)
        if not market_open:
            return

        # Iterate over each stock symbol

        option_positions = self.GetOptionPositions(self.spy)
        unrealized_profit = 0.0
        investment = 0.0
        for position in option_positions:
            unrealized_profit += position.Value.UnrealizedProfit
            investment += abs(position.Value.Quantity) * position.Value.AveragePrice
        if investment > 0:
            unrealized_profit_percent = unrealized_profit / investment
            if unrealized_profit_percent <= self.stop_loss_pct:
                #self.Log(
                #    f"Stop loss triggered, liquidating positions in {self.spy.ToString()}. Currently unrealized P&L %: {unrealized_profit_percent}"
                #)
                self.Liquidate(self.spy)

    def GetOptionPositions(self, symbol):
        return [
            x
            for x in self.Portfolio
            if x.Value.Invested
            and x.Value.Type == SecurityType.Option
            and x.Value.Symbol.Underlying == symbol
        ]
    

    def PrepareData(self):

        # Ensure data lists are non-empty
        if len(self.sp500_data) == 0 or len(self.vix_data) == 0 or len(self.indicators_data) == 0:
            self.Debug("One or more data lists are empty!")
            return None, None

        # Ensure data lists have sufficient length
        min_length = min(len(self.sp500_data), len(self.vix_data), len(self.indicators_data))
        if min_length < self.data_buffer_size:
            self.Debug(f"Insufficient data: min_length = {min_length}, required = {self.data_buffer_size}")
            return None, None

        # Align data lists to the same length
        self.sp500_data = self.sp500_data[-min_length:]
        self.vix_data = self.vix_data[-min_length:]
        self.indicators_data = self.indicators_data[-min_length:]

        # Create DataFrame
        data = pd.DataFrame({
            'S&P 500': self.sp500_data,
            'VIX': self.vix_data,
            'MACD': [x[0] for x in self.indicators_data],
            'MACD Signal': [x[1] for x in self.indicators_data],
            'MACD Histogram': [x[2] for x in self.indicators_data],
            'Upper Band': [x[3] for x in self.indicators_data],
            'Lower Band': [x[4] for x in self.indicators_data]
        })

        # Calculate daily returns and realized volatility
        data['Daily Returns'] = pd.Series(self.sp500_data).pct_change().fillna(0)
        data['Realized Volatility'] = data['Daily Returns'].rolling(window=20).std() * np.sqrt(252) * 100

        # Drop NaN values
        data = data.dropna()
        if len(data) < 20:
            self.Debug(f"Not enough data after dropping NaNs: {len(data)}")
            return None, None

        # Scale features
        columns_to_scale = ['S&P 500', 'VIX', 'MACD', 'MACD Signal', 'MACD Histogram', 'Upper Band', 'Lower Band', 'Realized Volatility']
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[columns_to_scale])

        # Store the last 20 rows for predictions
        last_20_rows = data.iloc[-20:]
        self.latest = np.expand_dims(scaler.transform(last_20_rows[columns_to_scale]), axis=0)

        # Create target variable
        data['Target'] = (data['Realized Volatility'].shift(-1) > data['Realized Volatility']).astype(int)

        # Drop NaN values in the target column
        data = data.dropna()
        if len(data) < 30:
            self.Debug(f"Not enough data for training after target creation: {len(data)}")
            return None, None

        # Prepare LSTM sequences
        sequence_length = 10
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(scaled_data[i-sequence_length:i])
            y.append(data['Target'].iloc[i])

        return np.array(X), np.array(y)


    def train_test_LSTM(self, X_train, X_val, X_test, Y_train, Y_val, Y_test):
        # Create and compile your LSTM model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.LSTM_Algo.LSTM_model(input_shape)

        # Train the model with validation data
        model.fit(X_train, Y_train, epochs=self.LSTM_Algo.epochs, batch_size=self.LSTM_Algo.batch_size, validation_data=(X_val, Y_val))

        # After training, evaluate the model on the test set
        test_loss, test_accuracy = model.evaluate(X_test, Y_test)
        return model, test_accuracy

    def TrainTestSplit(self, X, y, test_size=0.2, validation_size=0.1):
        # Split into training and testing data
        split_index = int(len(X) * (1 - test_size))
        X_train, X_temp = X[:split_index], X[split_index:]
        y_train, y_temp = y[:split_index], y[split_index:]

        # Further split the temporary data into validation and test
        validation_index = int(len(X_temp) * (1 - validation_size))
        X_val, X_test = X_temp[:validation_index], X_temp[validation_index:]
        y_val, y_test = y_temp[:validation_index], y_temp[validation_index:]

        # Return train, validation, and test sets
        return X_train, X_val, X_test, y_train, y_val, y_test


