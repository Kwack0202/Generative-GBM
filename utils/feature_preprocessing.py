from common_imports import *

# 기술적 지표 생성 함수
def calculate_indicators(df):
       
    # APO
    df['APO'] = talib.APO(df['Adj Close'], fastperiod=12, slowperiod=26, matype=0)

    # CMO (Chande Momentum Oscillator)
    df['CMO'] = talib.CMO(df['Adj Close'], timeperiod=14)

    # MACCD
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Adj Close'], fastperiod = 12, slowperiod = 26, signalperiod = 9)
    
    # MOM (Momentum)
    df['MOM'] = talib.MOM(df['Adj Close'], timeperiod=10)

    # PPO (Percentalibge Price Oscillator)
    df['PPO'] = talib.PPO(df['Adj Close'], fastperiod=12, slowperiod=26, matype=0)

    # ROC (Rate of Change)
    df['ROC'] = talib.ROC(df['Adj Close'], timeperiod=10)

    # ROCR (Rate of Change Ratio)
    df['ROCR'] = talib.ROCR(df['Adj Close'], timeperiod=10)

    # RSI (Relative Strength Index)
    df['RSI'] = talib.RSI(df['Adj Close'], timeperiod=14)

    # STOCHRSI (Stochastic Relative Strength Index)
    df['STOCHRSI_fastk'], df['STOCHRSI_fastd'] = talib.STOCHRSI(df['Adj Close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)

    # TRIX (1-day Rate of Change of a Triple Smooth EMA)
    df['TRIX'] = talib.TRIX(df['Adj Close'], timeperiod=30)
    
    return df

# 라벨 생성 함수
def add_labels(df):
    df['Up_Down'] = np.where((df['Adj Close'].shift(-1) - df['Adj Close']) / df['Adj Close'] >= 0.00, 1, 0)
    return df
