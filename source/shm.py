import numpy as np
import pandas as pd

def shm_gen(dt=0.001,
            coef=100,  # coef = k/m
            amplitude=2,
            start_trend=100,
            trend_per_tick=0.0,
            noise=0.0,
            damping=0.0,
            verbose=False):
    """Generate simple harmonic motion around trend, with noise and damping"""

    period = 2 * np.pi * np.sqrt(1 / coef)

    if verbose:
        print("%s Amplitude: %.3f" % (time.strftime("%H:%M:%S"), amplitude))
        print("%s Period: %.3f" % (time.strftime("%H:%M:%S"), period))

    # initial stock price
    stock_price = start_trend + amplitude
    stock_velocity = 0.0

    trend_index = start_trend
    t = 0.0

    while True:
        # acceleration based on distance from trend
        acc = - coef * (stock_price - trend_index)
        stock_velocity += acc * dt
        # add noise to velocity
        stock_velocity += np.random.normal(loc=0, scale=noise)
        # damp velocity by a % (could also make this a constant)
        stock_velocity *= (1 - damping)
        # increment stock price
        stock_price += stock_velocity * tick_length
        # add noise; doesn't impact velocity which makes velocity a partly hidden state variable
        stock_price += np.random.normal(loc=0, scale=noise / 2)

        yield (t, stock_price, trend_index)
        t += dt

if __name__ == "__main__":
    # simulate market data
    total_time = 1
    ticks = 10000
    tick_length = total_time / ticks

    # coef = k/m
    coef = 100
    amplitude = 2
    start_trend = 100
    trend_per_tick = 0.0
    noise = 0.0
    damping = 0.0

    period = 2 * np.pi * np.sqrt(1 / coef)
    print(period)
    # gen = shm_gen(dt=total_time/ticks,
    #               coef=coef,
    #               amplitude=amplitude,
    #               start_trend=start_trend,
    #               trend_per_tick=trend_per_tick,
    #               noise=noise,
    #               damping=damping,
    #               verbose=1)

    gen = shm_gen()

    trend_series = []
    stock_series = []
    time_series = []

    for i in range(ticks):
        t, stock_price, trend_index = next(gen)
        stock_series.append(stock_price)
        trend_series.append(trend_index)
        time_series.append([str(i), stock_price, stock_price, stock_price, trend_index])
    df = pd.DataFrame(time_series, columns=['날짜','종가', '저가', '고가', '거래량'])
    df.to_csv('/Users/user/projects/DLBollingerTrader_dev/trade_rl/data/train_dummy/shm.tsv', sep='\t', index=False)