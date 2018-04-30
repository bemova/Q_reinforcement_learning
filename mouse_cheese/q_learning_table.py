import numpy as np
import pandas as pd


class QLearning:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, df=None):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        # df is a data frame of pretrained Q table
        if df is None:
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        else:
            self.q_table = df

    def choose_action(self, observation):
        self.check_state_exist(observation)
        rand = np.random.uniform()
        if rand < self.epsilon:
            # this case we have to select the beast action based on Q table
            # which is the action that has the max value
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            #  in this case we let the agent to have some exploration on the environment
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, next_state):
        self.check_state_exist(next_state)
        predict = self.q_table.ix[s, a]
        if next_state != 'terminal':
            target = r + self.gamma * self.q_table.ix[next_state, :].max()
        else:
            target = r
        self.q_table.ix[s, a] += self.lr * (target - predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index.astype(str):
            zero_series = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            self.q_table = self.q_table.append(zero_series)
