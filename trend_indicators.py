from math import sqrt


def simple_moving_average(df, base, target, period):
    """
    Function to compute Simple Moving Average (SMA)
    This is a lagging indicator
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    target - column name to store output
    period - period of the sma
    """
    df[target] = df[base].rolling(window=period).mean().round(2)
    return df


def exponential_moving_average(df, base, target, period):
    """
    Function to compute Exponential Moving Average (EMA)
    This is a lagging indicator
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    target - column name to store output
    period - period of the ema
    """
    df[target] = df[base].ewm(ignore_na=False, min_periods=period, com=period, adjust=True).mean()
    return df


def moving_average_convergence_divergence(df, base, macd_target, macd_line_target, period_long=26, period_short=12,
                                          period_signal=9):
    """
    Function to compute MACD (Moving Average Convergence/Divergence)
    This is a lagging indicator
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    macd_target - column name to store macd value
    macd_line_target - column name to store macd line
    period_long - period of the longer time frame
    period_short - period of the shorter time frame
    period_signal - period of the signal
    """
    short_ema_target = 'ema_{}'.format(period_short)
    long_ema_target = 'ema_{}'.format(period_long)

    df = exponential_moving_average(df, base, long_ema_target, period_long)
    df = exponential_moving_average(df, base, short_ema_target, period_short)

    df[macd_target] = df[short_ema_target] - df[long_ema_target]
    df[macd_line_target] = df[macd_target].ewm(ignore_na=False, min_periods=0, com=period_signal, adjust=True).mean()
    df = df.drop([short_ema_target, long_ema_target], axis=1)
    return df


def linear_regression(df, base, period=200):
    """
    Function to compute Linear Regression Channels
    df - the data frame
    base - on which the indicator has to be calculated eg Close
    period
    """

    close_prices = df[base][-period:]
    if len(close_prices) < period:
        return None

    sum_x = 0.0
    sum_y = 0.0
    sum_x_sqr = 0.0
    sum_xy = 0.0

    i = 0
    for s in close_prices:
        per = i + 1.0
        sum_x = sum_x + per
        sum_y = sum_y + s
        sum_x_sqr = sum_x_sqr + per * per
        sum_xy = sum_xy + s * per
        i = i + 1

    slope = (period * sum_xy - sum_x * sum_y) / (period * sum_x_sqr - sum_x * sum_x)
    intercept = sum_y / period - slope * sum_x / period + slope

    std_dev_acc = 0.0
    periods = period - 1
    val = intercept

    for s in close_prices:
        price = s - val
        std_dev_acc = std_dev_acc + price * price
        val = val + slope

    std_dev = sqrt(std_dev_acc / periods)
    linear_reg_line_end = round(intercept + slope * (period - 1), 2)
    linear_reg_line_start = round(intercept, 2)

    sd1_low = round(linear_reg_line_end - std_dev, 2)
    sd2_low = round(linear_reg_line_end - std_dev * 2, 2)
    sd3_low = round(linear_reg_line_end - std_dev * 3, 2)

    sd1_high = round(linear_reg_line_end + std_dev, 2)
    sd2_high = round(linear_reg_line_end + std_dev * 2, 2)
    sd3_high = round(linear_reg_line_end + std_dev * 3, 2)

    result = dict()
    result["start"] = linear_reg_line_start
    result["end"] = linear_reg_line_end
    result["sd1_low"] = sd1_low
    result["sd1_high"] = sd1_high
    result["sd2_low"] = sd2_low
    result["sd2_high"] = sd2_high
    result["sd3_low"] = sd3_low
    result["sd3_high"] = sd3_high

    return result


