import gym
import numpy as np
from rl4rs.server.gymHttpClient import Client


class HttpEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_id, config={}):
        remote_base = config["remote_base"]
        self.client = Client(remote_base)
        self.instance_id = self.client.env_create(env_id, config)
        action_info = self.client.env_action_space_info(self.instance_id)
        obs_info = self.client.env_observation_space_info(self.instance_id)
        if action_info['name'] == 'Box':
            self.action_space = gym.spaces.Box(np.array(action_info['low']), np.array(action_info['high']), shape=action_info['shape'])
        else:
            self.action_space = gym.spaces.Discrete(action_info['n'])
        if obs_info['name'] == 'Box':
            self.observation_space = gym.spaces.Box(np.array(obs_info['low']), np.array(obs_info['high']), shape=obs_info['shape'])
        elif obs_info['name'] == 'Dict':
            keys = obs_info['keys']
            space_D = {}
            for key in keys:
                shape = obs_info[key]['shape']
                space_D[key] = gym.spaces.Box(np.array(obs_info[key]['low']).reshape(shape), np.array(obs_info[key]['high']).reshape(shape), shape=shape)
            self.observation_space = gym.spaces.Dict(space_D)
        else:
            assert obs_info['name'] in ('Box', 'Dict')

    def seed(self, sd=0):
        pass

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, np.int):
            action = int(action)
        observation, reward, done, info = self.client.env_step(self.instance_id, action, False)
        return self.observation_space.from_jsonable(observation), reward, done, info

    def reset(self):
        observation = self.client.env_reset(self.instance_id)
        return self.observation_space.from_jsonable(observation)

    def render(self, mode='human', close=False):
        return ''

    def close(self):
        return self.client.env_close(self.instance_id)
