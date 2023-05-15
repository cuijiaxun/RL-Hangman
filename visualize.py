import hydra
import torch
import numpy as np
from rlmeta.core.types import Action
from env import HangmanEnv
from models import PPOTransformerModel
from utils.gym_wrappers import GymWrapper

@hydra.main(config_path="config", config_name="hangman")
def main(cfg):
    model = PPOTransformerModel(**cfg.model_config).to(cfg.train_device)
    checkpoint = torch.load("/home/cuijiaxun/Projects/RL-Hangman/outputs/2023-05-15/18-26-22/ppo_agent-70.pth")
    model.load_state_dict(checkpoint)
    env = HangmanEnv(cfg.env_config)
    env = GymWrapper(env)
    success = []
    for i in range(1000):
        timestep = env.reset()
        obs = timestep.observation
        done = False
        while not done:
            timestep = env.step(Action(model.act(obs.unsqueeze(0), deterministic_policy=torch.tensor([True]))[0],{}))
            obs = timestep.observation
            done = timestep.done
            info = timestep.info
            print(info["state"])
        success.append(int(info["success"]))
        print(env.env.secret_word)
        print("Correct Rate:", np.mean(success))

if __name__ == "__main__":
    main()

