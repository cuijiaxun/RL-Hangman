import random
import numpy as np
import gym
import string
from collections import deque

class Hangman(gym.Env):
    def __init__(self, 
            max_life=6,
            action_space=26,
            state_space=32,
            history_length=26):

        with open("words_250000_train.txt",'r') as f:
            self.wordlist = f.read().splitlines()
        self.max_life = max_life
        self.num_life = max_life
        self.letters = string.ascii_lowercase
        self.action_space = action_space
        self.state_space = state_space
        self.history_length = history_length
        self.map=dict(zip(string.ascii_lowercase.split(), np.arange(1,26)))
        self.map.update({'_':27})

    def reset(self):
        self.secret_word = random.choice(self.wordlist)
        self.step = 0 
        self.len_secret_word = len(self.secret_word)
        self.state = ['_' for i in range(self.len_secret_word)]
        self.num_life = self.max_life
        self.action_mask = [True] * self.action_space
        self.obs = deque([[-1]*31]*self.history_length)
        self.current_obs = []
        for i in range(self.state_space):
            if i < len(self.state):
                self.current_obs.append(self.map[self.state[i]])
            elif i < self.state_space - 2:
                self.current_obs.append(-1)
            elif i < self.state_space - 1:
                self.current_obs.append(-1)
            else:
                self.current_obs.append(self.step)
        self.obs.append(self.current_obs)
        self.obs.popleft()
        done = False
        return np.array(list(reversed(self.obs))), done

    def step(self, action):
        matched = False
        reward = 0
        done = False
        info = {}
        
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
        
        return self.state, reward, done, info


if __name__ == "__main__":
    env = Hangman()
    obs, done = env.reset()
    
    print(obs)
    print("Please guess a letter:")
    while not done:
        print(env.action_mask)
        print("Remaining life:", env.num_life)
        next_state, reward, done, info = env.step(input())
        print(next_state)
    print(env.secret_word)
