def relative_strength_index(df, base, target, period=8):
    """
    Function to compute Relative Strength Index (RSI)
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    target - column name to store output
    period - period of the rsi
    """
    delta = df[base].diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    r_up = up.ewm(com=period - 1, adjust=False).mean()
    r_down = down.ewm(com=period - 1, adjust=False).mean().abs()

    df[target] = 100 - 100 / (1 + r_up / r_down)
    df[target].fillna(0, inplace=True)

    return df


def stochastic(df, low_base, high_base, close_base, target, period=8):
    """
    Stochastic Oscillator
    df - the data frame
    low_base - low price column
    high_base - high price column
    close_base - close price column
    target - column name to store output
    period - period of the stochastic
    """
    smin = df[low_base].rolling(period, min_periods=0).min()
    smax = df[high_base].rolling(period, min_periods=0).max()

    df[target] = 100 * (df[close_base] - smin) / (smax - smin)
    df[target] = df[target].round(2)
    df[target].fillna(0, inplace=True)

    return df
