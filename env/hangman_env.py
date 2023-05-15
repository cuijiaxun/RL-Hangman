import random
import numpy as np
import gym
from gym import spaces
import string
from collections import deque

class HangmanEnv(gym.Env):
    def __init__(self,
            env_config={
                "max_life":6,
                "action_space":26,
                "state_space":32,
                "history_length":26}):

        with open("/home/cuijiaxun/Projects/RL-Hangman/env/words_250000_train.txt",'r') as f:
            self.wordlist = f.read().splitlines()
        self.max_life = env_config.max_life
        self.num_life = env_config.max_life
        self.letters = string.ascii_lowercase
        self.action_space = spaces.Discrete(env_config.action_space)
        self.state_space = env_config.state_space
        self.history_length = env_config.history_length
        self.map=dict(zip([*string.ascii_lowercase], np.arange(1,27)))
        self.map.update({'_':27})
        self.reverse_map=dict(zip(np.arange(0,26), [*string.ascii_lowercase]))
        self.observation_space = spaces.Box(low=-1, high=28, shape=(self.history_length, 32))

    def reset(self):
        self.secret_word = random.choice(self.wordlist)
        self.step_count = 0 
        self.len_secret_word = len(self.secret_word)
        self.state = ['_' for i in range(self.len_secret_word)]
        self.num_life = self.max_life
        self.action_mask = [True] * self.action_space.n
        self.obs = deque([[-1]*32]*self.history_length)
        self.current_obs = []
        for i in range(self.state_space):
            if i < len(self.state):
                self.current_obs.append(self.map[self.state[i]])
            elif i < self.state_space - 2:
                self.current_obs.append(-1)
            elif i < self.state_space - 1:
                self.current_obs.append(-1)
            else:
                self.current_obs.append(self.step_count)
        self.obs.append(self.current_obs)
        self.obs.popleft()
        done = False
        return np.array(list(reversed(self.obs)), dtype=np.float32)

    def step(self, action):
        action = self.reverse_map[int(action)]
        matched = False
        reward = 0
        done = False
        info = {}
        self.step_count += 1

        if action in self.secret_word:
            for i, letter in enumerate(self.secret_word):
                if action == letter:
                    self.state[i] = action
                    matched = True

        if not matched:
            self.num_life -= 1
            if self.num_life == 0:
                reward = -1
                done = True
        
        if ''.join(self.state) == self.secret_word:
            reward = 1
            done = True
        
        self.action_mask[self.letters.find(action)] = False
        
        self.current_obs = []
        for i in range(self.state_space):
            if i < len(self.state):
                self.current_obs.append(self.map[self.state[i]])
            elif i < self.state_space - 2:
                self.current_obs.append(-1)
            elif i < self.state_space - 1:
                self.current_obs.append(self.map[action])
            else:
                self.current_obs.append(self.step_count)
        self.obs.append(self.current_obs)
        self.obs.popleft()
      
        return np.array(list(reversed(self.obs)),dtype=np.float32), reward, done, info


if __name__ == "__main__":
    env = HangmanEnv()
    
    for i in range(2):
        obs, done = env.reset()
        print(obs)
        print("Please guess a letter:")
        while not done:

            from IPython import embed; embed()
            print(env.action_mask)
            print("Remaining life:", env.num_life)
            next_state, reward, done, info = env.step(input())
            print(next_state, reward)
        print(env.secret_word)
