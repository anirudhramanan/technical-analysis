from math import sqrt

from trend_indicators import exponential_moving_average


def average_true_range(df, target, period):
    """
    Function to compute Average True Range (ATR)
    This is a lagging indicator
    df - the data frame
    target - column name to store output
    period - period of the atr
    """
    base = 'TR'
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift())
    df['L-PC'] = abs(df['Low'] - df['Close'].shift())
    df[base] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)

    df = exponential_moving_average(df, base, target, period)
    df = df.drop(['H-L', 'H-PC', 'L-PC', base], axis=1)
    return df


def bollinger_band(df, base, upper_target, lower_target, period):
    """
    Function to compute Bollinger Bands (BB)
    This is a lagging indicator
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    upper_target - column name to store upper BB value
    lower_target - column name to store lower BB value
    period - period of the bb
    """
    df['{}MA'.format(period)] = df[base].rolling(window=period).mean()
    df['{}STD'.format(period)] = df[base].rolling(window=period).std()
    df[upper_target] = df['{}MA'.format(period)] + (df['{}STD'.format(period)] * 2)
    df[lower_target] = df['{}MA'.format(period)] - (df['{}STD'.format(period)] * 2)

    df = df.drop(['{}MA'.format(period), '{}STD'.format(period)], axis=1)
    return df


def bollinger_bandwidth(df, base, upper_target, lower_target, target, period):
    """
    Function to compute Bollinger BandWidth (BB)
    This is a lagging indicator
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    upper_target - column name to store upper BB value
    lower_target - column name to store lower BB value
    target - column name to store result
    period - period of the bb
    """
    df['{}MA'.format(period)] = df[base].rolling(window=period).mean()
    df['{}STD'.format(period)] = df[base].rolling(window=period).std()
    df[upper_target] = df['{}MA'.format(period)] + (df['{}STD'.format(period)] * 2)
    df[lower_target] = df['{}MA'.format(period)] - (df['{}STD'.format(period)] * 2)

    df[target] = ((df[upper_target] - df[lower_target]) / df['{}MA'.format(period)]) * 100
    df = df.drop(['{}MA'.format(period), '{}STD'.format(period)], axis=1)
    return df


def historical_volatility(df, base, period=40):
    """
    Function to compute historical volatility
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    period - period
    """
    close_prices = df[base][-period:]
    avg_price = sum(close_prices) / len(close_prices)
    squared_diff_prices = []

    for close_price in close_prices:
        diff = close_price - avg_price
        squared_diff_prices.append(diff * diff)

    sum_squared_diff_prices = sum(squared_diff_prices)
    variance = sum_squared_diff_prices / period
    st_deviation = sqrt(variance)

    return st_deviation
