import numpy as np
from typing import Callable, List, Optional, Tuple
from ray.rllib.utils.typing import EnvActionType, EnvConfigDict, EnvInfoDict, \
    EnvObsType, EnvType, PartialTrainerConfigDict
from ray.rllib.env.vector_env import VectorEnv
from rl4rs.env import RecEnvBase


class MyVectorEnvWrapper(VectorEnv):
    """An environment that supports batch evaluation using clones of sub-envs.
    """

    def __init__(self, env: RecEnvBase, batch_size: int):
        """Initializes a VectorEnv object.

        Args:
            observation_space (Space): The observation Space of a single
                sub-env.
            action_space (Space): The action Space of a single sub-env.
            num_envs (int): The number of clones to make of the given sub-env.
        """
        self.env = env
        self.reset_cache = []
        super().__init__(self.env.observation_space, self.env.action_space, num_envs=batch_size)

    def vector_reset(self) -> List[EnvObsType]:
        """Resets all sub-environments.

        Returns:
            obs (List[any]): List of observations from each environment.
        """
        return self.env.reset()

    def reset_at(self, index: Optional[int] = None) -> EnvObsType:
        """Resets a single environment.

        Args:
            index (Optional[int]): An optional sub-env index to reset.

        Returns:
            obs (obj): Observations from the reset sub environment.
        """
        if index == 0:
            self.reset_cache = self.env.reset()
        return self.reset_cache[index]

    def vector_step(
            self, actions: List[EnvActionType]
    ) -> Tuple[List[EnvObsType], List[float], List[bool], List[EnvInfoDict]]:
        """Performs a vectorized step on all sub environments using `actions`.

        Args:
            actions (List[any]): List of actions (one for each sub-env).

        Returns:
            obs (List[any]): New observations for each sub-env.
            rewards (List[any]): Reward values for each sub-env.
            dones (List[any]): Done values for each sub-env.
            infos (List[any]): Info values for each sub-env.
        """
        return self.env.step(np.array(actions))

    def get_unwrapped(self) -> List[EnvType]:
        """Returns the underlying sub environments.

        Returns:
            List[Env]: List of all underlying sub environments.
        """
        return [self.env, ] * self.num_envs

    # Experimental method.
    def try_render_at(self, index: Optional[int] = None) -> None:
        """Renders a single environment.

        Args:
            index (Optional[int]): An optional sub-env index to render.
        """
        return self.env.render()
