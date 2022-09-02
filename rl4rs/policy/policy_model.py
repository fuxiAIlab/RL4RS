import torch
import d3rlpy
import numpy as np
from ray.rllib.agents.trainer import Trainer as rllib_trainer
from scipy.special import softmax


class policy_model(object):
    def __init__(self, model, config = {}):
        self.policy = model
        self.config = config
        self.page_items = int(config.get('page_items', 9))
        self.mask_size = self.page_items+1
        self.location_mask = config.get('location_mask', None)
        self.special_items = config.get('special_items', None)

    def predict_with_mask(self, obs):
        if self.config.get("support_conti_env",False):
            return self.predict(obs)
        elif isinstance(self.policy, d3rlpy.algos.AlgoBase):
            obs = np.array(obs)
            action_probs = np.array(self.action_probs(obs))
            batch_size = len(obs)
            # mask
            prev_actions = obs[:, -self.mask_size:-1].astype(int)
            cur_step = obs[:, -1].astype(int)
            x_mask_layer = cur_step % self.page_items // 3
            mask = self.location_mask[x_mask_layer.astype(int)]
            for i in range(self.mask_size-1):
                mask[range(batch_size), prev_actions[:, i]] = 0
            action_mask = mask < 0.01
            action_probs[action_mask] = -2 ** 15
            for i in range(batch_size):
                if len(np.intersect1d(prev_actions[i], self.special_items)) > 0:
                    action_probs[i][self.special_items] = -2 ** 15
            return action_probs.argmax(axis=1)
        elif isinstance(self.policy, rllib_trainer):
            return self.predict(obs)
        else:
            assert isinstance(self.policy, d3rlpy.algos.AlgoBase) \
                   or isinstance(self.policy, rllib_trainer)

    def predict(self, obs):
        if isinstance(self.policy, d3rlpy.algos.AlgoBase):
            return self.policy.predict(obs)
        elif isinstance(self.policy, rllib_trainer):
            obs = dict(enumerate(obs))
            action = self.policy.compute_actions(obs, explore=False)
            action = np.array(list(action.values()))
            return action
        else:
            assert isinstance(self.policy, d3rlpy.algos.AlgoBase) \
                   or isinstance(self.policy, rllib_trainer)

    def predict_q(self, obs, action):
        if isinstance(self.policy, d3rlpy.algos.AlgoBase):
            q = self.policy.predict_value(obs, action)
            if self.policy.reward_scaler is not None:
                return self.policy.reward_scaler.reverse_transform(q)
            else:
                return q
        elif isinstance(self.policy, rllib_trainer):
            obs = dict(enumerate(obs))
            _, _, infos = self.policy. \
                compute_actions(obs, explore=False, full_fetch=True)
            batch_size = len(action)
            return infos['q_values'][range(batch_size), action] \
                if 'q_values' in infos \
                else infos['vf_preds']
        else:
            assert isinstance(self.policy, d3rlpy.algos.AlgoBase) \
                   or isinstance(self.policy, rllib_trainer)

    def action_probs(self, obs):
        if isinstance(self.policy, d3rlpy.algos.DiscreteBC):
            obs = torch.tensor(obs, dtype=torch.float32)
            return self.policy._impl._imitator(obs).detach().numpy()
        elif isinstance(self.policy, d3rlpy.algos.DiscreteBCQ) \
                or isinstance(self.policy, d3rlpy.algos.DiscreteCQL):
            obs = torch.tensor(obs, dtype=torch.float32)
            action_q = self.policy._impl._q_func(obs).detach().numpy()
            return softmax(action_q, axis=1)
        elif isinstance(self.policy, rllib_trainer):
            obs = dict(enumerate(obs))
            actions, _, infos = self.policy. \
                compute_actions(obs, explore=False, full_fetch=True)
            return softmax(infos['action_dist_inputs'], axis=1)
        else:
            assert isinstance(self.policy, d3rlpy.algos.AlgoBase) \
                   or isinstance(self.policy, rllib_trainer)
