

class Env():
    def __init__(self, stock_df, stock_name, window):
        self.idx = -1
        self.epoch = 1
        self.stock_name = stock_name
        self.window = window
        self.done = False
        self.stock_df = stock_df
        self.model_state = None
        self.model_next_state = None
        for i in range(window+1):
            self.step()


    def step(self):
        self.idx += 1
        if self.idx >= (len(self.stock_df) - 1): self.done = True
        else: self.done = False
        self.stock_state = self.stock_df.iloc[self.idx]
        if not self.done:
            self.stock_next_state = self.stock_df.iloc[self.idx+1]
        else: self.stock_next_state = None

    def reset(self):
        self.idx = -1
        for i in range(self.window+1):
            self.step()








