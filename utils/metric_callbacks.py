from rlmeta.core.callbacks import EpisodeCallbacks
from rlmeta.core.types import Action, TimeStep


class MetricCallbacks(EpisodeCallbacks):
    def __init__(self):
        super().__init__()

    def on_episode_step(self, index: int, step: int, action: Action,
                        timestep: TimeStep) -> None:
        info = timestep.info