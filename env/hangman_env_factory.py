import os
import sys

from typing import Any, Dict

from rlmeta.envs.env import Env, EnvFactory
from utils.gym_wrappers import GymWrapper

from .hangman_env import HangmanEnv


class HangmanEnvFactory(EnvFactory):
    def __init__(self, env_config: Dict[str, Any]) -> None:
        self._env_config = env_config

    @property
    def env_config(self) -> Dict[str, Any]:
        return self._env_config

    def __call__(self, index: int, *args, **kwargs) -> Env:
        env = HangmanEnv(self.env_config)
        env = GymWrapper(env)
        return env
