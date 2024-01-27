import pandas as pd
import numpy as np

class PriceUtil():
    def __init__(self):
        pass

    # n 일에 대한 m 구간 볼린져 밴드 정보 df 에 부착
    # in : df(panda Data Frame), n(이동 평균 일수), m(표준 편차 사이즈)
    # out : 아래 필드 부착
    #          - B_MA_n : n 일 이동평균
    #          - B_U_n_m : n 일 이동평균에 대한 m 표준편차 Upper Band
    #          - B_L_n_m : n 일 이동평균에 대한 m 표준편차 Lower Band
    def add_bollinger_bands(self, df, n, m):
        # n = smoothing length
        # m = number of standard deviations away from MA
        TP = (df['고가'] + df['저가'] + df['종가']) / 3
        B_MA = pd.Series((TP.rolling(n, min_periods=n).mean()), name='B_MA_%d' % (n))
        sigma = TP.rolling(n, min_periods=n).std()
        BU = pd.Series((B_MA + m * sigma), name='B_U_%d_%d' % (n, m))
        BL = pd.Series((B_MA - m * sigma), name='B_L_%d_%d' % (n, m))
        df = df.join(B_MA)
        df = df.join(BU)
        df = df.join(BL)
        df["B_PB_%d_%d" % (n, m)] = (df['종가'] - BL / (BU - BL))
        return df

    def add_mfi(self, df, period):
        typical_price = (df['종가'] + df['고가'] + df['저가']) / 3
        money_flow = typical_price * df['거래량']
        positive_flow = []
        negative_flow = []

        # Loop through the typical price
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i - 1]:
                positive_flow.append(money_flow[i - 1])
                negative_flow.append(0)

            elif typical_price[i] < typical_price[i - 1]:
                negative_flow.append(money_flow[i - 1])
                positive_flow.append(0)

            else:
                positive_flow.append(0)
                negative_flow.append(0)

        positive_mf = []
        negative_mf = []

        for i in range(period - 1, len(positive_flow)):
            positive_mf.append(sum(positive_flow[i + 1 - period: i + 1]))

        for i in range(period - 1, len(negative_flow)):
            negative_mf.append(sum(negative_flow[i + 1 - period: i + 1]))
        tail = (100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf))))
        tail = [t*0.01 for t in tail]
        df['mfi_%s' % str(period)] = [None]*period + list(tail)
        return df

    def min_max_normalize(self, df):
        min_price = min(df['종가']+df['고가']+df['저가'])
        denom = (max(df['종가']+df['고가']+df['저가']) - min_price)
        df['종가'] = (df['종가'] - min_price) / denom
        df['고가'] = (df['고가'] - min_price) / denom
        df['저가'] = (df['저가'] - min_price) / denom
        return df

    def add_rsi(self, df, period):
        change = df["종가"].diff()
        change.dropna(inplace=True)
        # Create two copies of the Closing price Series
        change_up = change.copy()
        change_down = change.copy()

        change_up[change_up < 0] = 0
        change_down[change_down > 0] = 0

        # Verify that we did not make any mistakes
        change.equals(change_up + change_down)

        # Calculate the rolling average of average up and average down
        avg_up = change_up.rolling(period).mean()
        avg_down = change_down.rolling(period).mean().abs()
        df['rsi_%d' % (period)] = avg_up / (avg_up + avg_down)
        return df