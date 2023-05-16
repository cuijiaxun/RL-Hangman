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

        with open("/u/jxcui/Documents/Projects/RL-Hangman/env/words_250000_train.txt",'r') as f:
            self.wordlist = f.read().splitlines()
        self.max_life = 6 #env_config.max_life or 6
        self.num_life = 6 #env_config.max_life or 6
        self.letters = string.ascii_lowercase
        self.action_space = spaces.Discrete(26) #env_config.action_space or 26)
        self.state_space = 32 #env_config.state_space or 32
        self.history_length = 26 #env_config.history_length or 26
        self.map=dict(zip([*string.ascii_lowercase], np.arange(0,26)))
        self.map.update({'_':26})
        self.reverse_map=dict(zip(np.arange(0,26), [*string.ascii_lowercase]))
        self.observation_space = spaces.Box(low=-1, high=28, shape=(self.history_length, 32))
        self.max_episode_length = 64

    def reset(self):
        self.secret_word = random.choice(self.wordlist)
        self.step_count = 0 
        self.len_secret_word = len(self.secret_word)
        self.state = ['_' for i in range(self.len_secret_word)]
        self.num_life = self.max_life
        self.action_mask = [True] * self.action_space.n
        self.guessed = np.zeros(self.action_space.n)
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
        info = {}
        info["action_mask"] = self.guessed
        return self.one_hot(np.array(list(reversed(self.obs)), dtype=int)), info

    def step(self, action):
        reward = 0
        if int(action) < 0 or int(action) >= 26:
            print("!!!!!!!", action)
        if self.guessed[int(action)] == 1:
            reward = -0.00
        self.guessed[int(action)] = 1
        action = self.reverse_map[int(action)]
        matched = False
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
                reward = -1.0
                done = True
                info["success"] = False
        #else:
        #    reward += 0.01

        if ''.join(self.state) == self.secret_word:
            reward = 1
            done = True
            info["success"] = True
        
        if self.step_count >= self.max_episode_length:
            done = True
            if "success" in info.keys():
                info["success"]=info["success"]
            else:
                info["success"]=False
                reward = -1.0
        self.action_mask[self.letters.find(action)] = False
        info["action_mask"] = self.guessed 
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
        info["state"] = self.state 
        return self.one_hot(np.array(list(reversed(self.obs)),dtype=int)), reward, done, info

    def one_hot(self, obs):
        '''
        total_obs = []
        for i in range(obs.shape[0]):
            curr_encode = []
            for j in range(obs.shape[1]-2):
                code = np.zeros(28)
                code[obs[i][j]+1] = 1
                curr_encode.extend(list(code))
            j = obs.shape[1]-2
            code = np.zeros(26)
            code[obs[i][j]] = 1
            curr_encode.extend(list(code))
            curr_encode.extend([obs[i][-1]])
            total_obs.append(curr_encode)
        '''
        total_obs = np.zeros((30,55))
        for i in range(total_obs.shape[0]):
            if i < len(self.state):
                letter = self.state[i]
                total_obs[i][self.map[letter]] = 1
                total_obs[i][27]=(i+1)#/30
            total_obs[i][28:54]=self.guessed
            total_obs[i][54]=self.num_life#/6
        return np.array(total_obs, dtype=np.float32)

if __name__ == "__main__":
    env = HangmanEnv(env_config={"max_life":6,
                "action_space":26,
                "state_space":32,
                "history_length":26}
            )
    
    for i in range(2):
        obs = env.reset()
        done = False
        print(obs)
        print("Please guess a letter:")
        while not done:
            print(obs.shape)
            print(env.action_mask)
            print("Remaining life:", env.num_life)
            next_state, reward, done, info = env.step(input())
            print(next_state, reward)
        print(env.secret_word)
