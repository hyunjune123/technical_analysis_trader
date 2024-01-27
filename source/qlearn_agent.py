import random
import numpy as np
import pandas as pd
import torch

class Agent:
    def __init__(self, model):
        # input parameter
        self.prev = None

        # hyper parameter
        self.epoch = 1
        self.g = 0.5 # discount rate
        self.lr = 0.5 # learning rate for q learning
        self.epsilon = 1 # greedy-epsilon
        self.min_epsilon = 0.01
        self.decay = 0.995 #0.995 # epsilon will decay overtime
        self.model = model # Qvalue NN approximator
        self.memory = pd.DataFrame(columns = ["stock_state", "state", "price", "action", "next_state", "reward", "done"])
        self.memory_size = 10000
        self.sample_size = 256
        
        # agent parameter
        self.action_space = [0, 1] # {exit = 0 enter = 1} for now...
        self.positions = []
        self.inventory = 0

        # Makes decision based on NN if epsilon value is low enough
        # otherwise random decision
    def reset(self):
        self.positions = []
        self.inventory = 0

    def act(self, env):
        def get_reward(action, env):
            def weighted_average(values):
                total_weight = sum(range(1, len(values) + 1))  # Total weight for normalization
                weighted_sum = sum([(i + 1) * value for i, value in enumerate(values)])
                weighted_avg = weighted_sum / total_weight
                return weighted_avg
            current_price = env.stock_state['종가']
            #next_price = env.stock_next_state['종가'] if not env.done else None
            if env.done: next_price = None
            else:
                next_prices = env.stock_df['종가'][(env.idx+1):(env.idx+11)]
                next_price = weighted_average(next_prices)
            if next_price:
                pchange = (next_price - current_price) / current_price * 100
            else:
                pchange = 0
            profitable = 10
            if env.done: reward = 0
            else:
                reward = (next_price - current_price) / current_price * 100
                #print("price_change : ", reward)
                if action == 1:
                    #reward = 0
                    pass
                    #print("buy : ", reward)
                    #print("enter reward (%f -> %f): " % (current_price, next_price), reward)
                else: # action == -1:
                    reward = -reward
                    #print("sell : ", reward)
                #else: #reward = 0
                #    if reward <= profitable and -profitable <= reward: reward = abs(reward)
                #    else: -1 * abs(reward)
            '''
            if not self.inventory:  # exit state
                if not action:  # remain exit
                    reward = 0
                else:  # enter
                    if not env.done:
                        reward = (next_price - current_price) / current_price

                    else:
                        reward = 0
                        #reward = (next_price - current_price) / current_price

            else:  # entered state
                if not action:  # exit
                    #reward = (current_price - self.inventory) / self.inventory
                    reward = 0
                else:  # remain enter
                    if not env.done:
                        #reward = (next_price - current_price) / self.inventory
                        reward = (next_price - current_price) / current_price
                        #reward = 1
                    else: # exit
                        #reward = (current_price - self.inventory) / self.inventory
                        reward = 0
            '''
            return reward, pchange
        state = env.model_state
        price = env.stock_state['종가']
        qvals = None
        # exploration
        if random.random() <= self.epsilon:
            typ = "explore"
            action = self.action_space[random.sample(self.action_space, 1)[0]]
        # exploitation
        else:
            typ = "exploit"
            with torch.no_grad():
                qvals = self.model.forward(torch.stack([state[0]]), torch.stack([state[1]])).numpy()[0]
                qvals = qvals.tolist()
            action = self.action_space[np.argmax(qvals)]
        assert action in [0, 1]
        self.positions.append(action)
        reward, pchange = get_reward(action, env)
        profit = 0
        if (action == 0 or env.done):
            if self.inventory:
                profit = (price-self.inventory)
            self.inventory = 0
        if action == 1:
            if not self.inventory:
                self.inventory = price
        #print(typ, action, qvals if typ == "exploit" else None)
        return typ, action, reward, profit, qvals, pchange

    # update agent's trade history memory
    # memory is used for training in an episode
    def update_memory(self, new_mem):
        if self.memory.shape[0] > self.memory_size:
            self.memory = self.memory[-self.memory_size:]
        #self.memory = pd.concat([self.memory, new_mem], ignore_index=True)
        self.memory.loc[len(self.memory)] = new_mem

    # Experience replay based update in Q-value NN approximator
    def update_agent_nn(self):
        # Not enough memory
        if len(self.memory) < self.sample_size:
            return
        minibatch = self.memory.sample(n=self.sample_size)
        '''
        pos_batch = (self.memory[self.memory['reward'] * self.memory['action'].apply(lambda x: 1 if x else -1) > 0])
        pos_batch = pos_batch.sample(n=min(len(pos_batch), self.sample_size//2))
        neg_batch = (self.memory[self.memory['reward'] * self.memory['action'].apply(lambda x: 1 if x else -1) < 0])
        neg_batch = neg_batch.sample(n=min(self.sample_size-len(pos_batch), len(neg_batch)))
        minibatch = pd.concat([pos_batch, neg_batch], ignore_index=True)
        '''
        X1, X2 = [], []
        for x in minibatch['state']:
            X1.append(x[0])
            X2.append(x[1])
        X_fit1 = torch.stack(X1)
        X_fit2 = torch.stack(X2)
        X_fit = list(zip(X_fit1, X_fit2))
        Y_pred = self.model.forward(X_fit1, X_fit2)
        Y_pred = Y_pred.detach().numpy()

        # if not done, add discount_rate  * Q-value for next state
        not_done = minibatch.loc[minibatch['done'] == False]
        nX1, nX2 = [], []
        for x in not_done['next_state']:
            nX1.append(x[0])
            nX2.append(x[1])
        nX1, nX2 = torch.stack(nX1), torch.stack(nX2)
        max_qs, _ = torch.max(self.model.forward(nX1, nX2), dim=1)

        # 1. MDP
        #  q(s, a) = reward + (g * maxa(Q(s')))
        '''
        minibatch['target_observed'] = minibatch['reward']
        next_q = [q * self.g for q in max_qs.tolist()]
        # update Q value for actions model took
        minibatch.loc[minibatch['done'] == False, 'target_observed'] += (next_q) # reward + g*maxa(Q(s'))
        np.put_along_axis(Y_pred,
                          (minibatch['action']).astype(int).values.reshape(self.sample_size, 1),
                          minibatch['target_observed'].values.reshape(self.sample_size, 1),
                          axis=1)
        '''

        # 2. Q value update
        # q(s, a) = lr * (reward + g * max(Q(s'))) + (1 - lr) * q(s, a)
        minibatch['next_q'] = minibatch['reward']
        next_q = [q * self.g for q in max_qs.tolist()]
        minibatch.loc[minibatch['done'] == False, 'next_q'] += next_q # reward + g * max(Q(s'))
        minibatch.loc[minibatch['done'] == False, 'next_q'] *= self.lr # lr * (reward + g * max(Q(s')))
        next_q = list(minibatch['next_q'])
        #print("nex_q : ", len(minibatch['next_q']))
        #print("y_pred : ", len(Y_pred))
        #print("actions : ", len(minibatch['action']))
        for i, action in enumerate(minibatch['action']):
            Y_pred[i][action] = (1-self.lr) * Y_pred[i][action] + next_q[i]
        batches = [(X_fit[i], Y_pred[i]) for i in range(len(X_fit))]
        loss = self.model.train(batches, bsize=32, epochs=1)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
        return loss