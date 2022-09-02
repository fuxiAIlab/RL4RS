import numpy as np
import gym
from copy import deepcopy as copy
import tensorflow as tf
from abc import ABC, abstractmethod
import os


def single_elem_support(func):
    """aop func"""
    type_list = (type([]), type(()), type(np.array(1)))

    def wrapper(*args, **kwargs):
        """wrapper func"""
        res = func(*args, **kwargs)
        if type(res) in type_list and len(res) == 1:
            return res[0]
        elif type(res[0]) in type_list and len(res[0]) == 1:
            return [x[0] for x in res]
        else:
            return res

    return wrapper


class RecState(ABC):
    def __init__(self, config, records):
        self.config = config
        self.records = records
        self._init_state = self.records_to_state(records)
        self._state = copy(self._init_state)

    @staticmethod
    def records_to_state(records):
        pass

    @property
    def state(self):
        return self._state

    @property
    @abstractmethod
    def user(self):
        pass

    @property
    @abstractmethod
    def info(self):
        pass

    @abstractmethod
    def act(self, actions):
        pass

    @abstractmethod
    def to_string(self):
        pass


class RecDataBase(object):
    '''
    file-based implementation of a RecommnedEnv's data source.

    Pulls data from file, preps for use by RecommnedEnv and then
    acts as data provider for each new episode.
    '''

    def __init__(self, config, state_cls):
        self.config = config
        self.sample_list = []
        self.state_cls = state_cls
        self.is_eval = config.get('is_eval', False)
        self.cache_size = config.get('cache_size', 2048)
        # sample file cache
        self.fp = open(config['sample_file'], 'r')
        # self.fp.readline()

    @staticmethod
    def seed(seed):
        np.random.seed(seed)

    def sample_cache(self, f, num):
        for i in range(num):
            tmp = f.readline().rstrip()
            if len(tmp) < 1:
                f.seek(0, 0)
                f.readline()
                self.sample_list.append(f.readline().rstrip())
            else:
                self.sample_list.append(tmp)

    def sample(self, batch_size):
        if self.is_eval:
            assert self.cache_size == batch_size
            assert len(self.sample_list) == batch_size
            records = self.sample_list[:batch_size]
        else:
            records = np.random.choice(self.sample_list, batch_size)
        samples = self.state_cls(self.config, records)
        return samples

    def reset(self, reset_file=False):
        # self.state_list = []
        self.sample_list = []
        # self.rawstate_cache(self.fs, 10000)
        if reset_file:
            self.fp.seek(0, 0)
        self.sample_cache(self.fp, self.cache_size)


class RecSimBase(ABC):
    """ Implemention of core recommendation simulator"""

    def __init__(self, config, state_cls):
        self.config = config
        self.max_steps = config['max_steps']
        self.batch_size = config['batch_size']
        model_file = config['model_file']
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.model = self.get_model(config)
            if self.config.get('gpu', False):
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
                self.sess = tf.Session(graph=self.graph,
                                       config=tf.ConfigProto(device_count={"CPU": 4}))
            else:
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                self.sess = tf.Session(graph=self.graph)
            self.saver = tf.train.Saver()
        self.reload_model(model_file)
        self._recData = RecDataBase(config, state_cls)

    def reset(self, reset_file=False):
        self._recData.reset(reset_file)

    @abstractmethod
    def get_model(self, config):
        pass

    @abstractmethod
    def obs_fn(self, state):
        pass

    @abstractmethod
    def forward(self, model, samples):
        pass

    def reload_model(self, model_file):
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver.restore(self.sess, model_file)

    def seed(self, sd=0):
        self._recData.seed(sd)
        np.random.seed(sd)

    def _step(self, samples, action, **kwargs):
        step = kwargs['step']
        samples.act(action)
        next_state = samples.state
        next_obs = self.obs_fn(next_state)
        reward = self.forward(self.model, samples)
        next_info = samples.info

        if step < self.max_steps - 1:
            done = [0] * self.batch_size
        else:
            done = [1] * self.batch_size

        return next_obs, reward, done, next_info

    def sample(self, batch_size):
        samples = self._recData.sample(batch_size)
        obs = self.obs_fn(samples.state)
        return samples, obs


class RecEnvBase(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, recsim: RecSimBase):
        self.config = recsim.config
        self.batch_size = self.config['batch_size']
        self.cur_step = 0
        self.sim = recsim
        self.sim.reset()
        self.samples, self.obs = self.sim.sample(self.batch_size)
        if self.config.get("rawstate_as_obs", False):
            category_size = len(self.obs[0]['category_feature'])
            dense_size = len(self.obs[0]['dense_feature'])
            sequence_size = np.array(self.obs[0]['sequence_feature']).shape
            features = {
                "category_feature": gym.spaces.Box(-1000000.0, 1000000.0, shape=(category_size,)),
                "dense_feature": gym.spaces.Box(-1000000.0, 1000000.0, shape=(dense_size,)),
                "sequence_feature": gym.spaces.Box(-1000000.0, 1000000.0, shape=sequence_size),
            }
            if self.config.get("support_rllib_mask", False):
                action_feature_size = len(self.obs[0]['action_mask'])
                self.observation_space = gym.spaces.Dict({
                    "action_mask": gym.spaces.Box(0, 1, shape=(action_feature_size,)),
                    **features
                })
            else:
                self.observation_space = gym.spaces.Dict(features)
        else:
            if self.config.get("support_rllib_mask", False):
                action_feature_size = len(self.obs[0]['action_mask'])
                self.observation_space = gym.spaces.Dict({
                    "action_mask": gym.spaces.Box(0, 1, shape=(action_feature_size,)),
                    "obs": gym.spaces.Box(-100000.0, 100000.0, shape=(len(self.obs[0]["obs"]),))
                })
            else:
                self.observation_space = gym.spaces.Box(-100000.0, 100000.0, shape=(len(self.obs[0]),))
        if self.config.get("support_conti_env", False):
            self.action_space = gym.spaces.Box(-1, 1, shape=(self.config['action_emb_size'],))
        else:
            self.action_space = gym.spaces.Discrete(self.config['action_size'])
        # if self.config.get("support_rllib_mask", False):
        #     action_feature_size = len(self.obs[0]['action_mask'])
        #     # avail_actions_size = len(self.obs[0]['avail_actions'][0])
        #     # self.action_space = gym.spaces.Discrete(self.config['action_size'])
        #     self.observation_space = gym.spaces.Dict({
        #         "action_mask": gym.spaces.Box(0, 1, shape=(action_feature_size,)),
        #         "obs": self.observation_space,
        #     })
        # elif self.config.get("support_d3rl_mask", False):
        #     self.action_space = gym.spaces.Discrete(self.config['action_size'])
        # else:
        #     self.action_space = gym.spaces.Discrete(self.config['action_size'])
        self.reset()

    def seed(self, sd=0):
        self.sim.seed(sd)
        np.random.seed(sd)

    @property
    @single_elem_support
    def state(self):
        return self.obs

    @property
    @single_elem_support
    def user_id(self):
        return self.samples.user

    @property
    @single_elem_support
    def offline_action(self):
        return self.samples.offline_action

    @property
    @single_elem_support
    def offline_reward(self):
        return self.samples.offline_reward

    @single_elem_support
    def step(self, action):
        if not isinstance(action, (list, np.ndarray)):
            action = [action]
        obs, reward, done, info = \
            self.sim._step(self.samples, action, step=self.cur_step)
        self.cur_step += 1
        return obs, reward, done, info

    def reset(self, reset_file=False):
        self.cur_step = 0
        self.sim.reset(reset_file)
        self.samples, self.obs = self.sim.sample(self.batch_size)
        return self.state

    def render(self, mode='human', close=False):
        print('Current State:', '\n')
        print(self.samples.to_string())
