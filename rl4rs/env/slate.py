import numpy as np
from copy import deepcopy as copy
from rl4rs.env import RecSimBase, RecState
import tensorflow as tf
from rl4rs.utils.datautil import FeatureUtil


class SlateState(RecState):
    def __init__(self, config, records):
        super().__init__(config, records)
        self.batch_size = self.config["batch_size"]
        self.action_size = self.config["action_size"]
        self.action_emb_size = self.config.get("action_emb_size", 32)
        self.max_steps = config['max_steps']
        self.infos = [{} for _ in range(self.batch_size)]
        self.prev_actions = np.full((self.batch_size, self.max_steps), 0)
        self.action_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)
        self.special_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)
        self.cur_steps = 0
        iteminfo_file = config["iteminfo_file"]
        self.item_info_d, self.action_emb = self.get_iteminfo_from_file(iteminfo_file, self.action_size)
        if config.get('support_onehot_action', False):
            config['action_emb_size'] = self.action_size
            self.action_emb_size = self.action_size
            self.action_emb = np.eye(self.action_size)
        self.location_mask, self.special_items = self.get_mask_from_file(iteminfo_file, self.action_size)

    @staticmethod
    def get_iteminfo_from_file(iteminfo_file, action_size, action_emb_size=32):
        item_info = open(iteminfo_file, 'r').read().split('\n')[1:]
        item_info = [x.split(' ') for x in item_info]
        item_info_d = dict(
            [(
                str(itemid),
                {
                    'item_vec': list(map(float, item_vec.split(','))),
                    'price': float(price),
                    'location': int(location)
                }
            ) for (itemid, item_vec, price, location, is_speical) in item_info]
        )
        item_info_d['0'] = {
            'item_vec': [0] * len(item_info_d['1']['item_vec']),
            'price': float(0),
            'location': int(0)
        }
        action_emb = np.zeros((action_size, action_emb_size))
        item_vecs = np.array([
            list(map(float, item_vec.split(',')))[-action_emb_size:]
            for (itemid, item_vec, price, location, is_speical) in item_info
        ])
        action_emb[1:] = np.einsum('ij,i->ij', item_vecs, 1.0 / np.linalg.norm(item_vecs, axis=1))
        return item_info_d, action_emb

    @staticmethod
    def get_mask_from_file(iteminfo_file, action_size):
        item_info = open(iteminfo_file, 'r').read().split('\n')[1:]
        item_info = [x.split(' ') for x in item_info]
        special_items = [int(itemid) for (itemid, item_vec, price, location, is_speical) in item_info if int(is_speical) == 2]
        location_mask = np.zeros((4, action_size), dtype=np.int)
        location_mask[0, 1:40] = 1
        location_mask[1, 40:148] = 1
        location_mask[2, 148:] = 1
        location_mask[3, 0] = 1
        return location_mask, special_items

    @staticmethod
    def records_to_state(records):
        def fn(record):
            _, _, sequence_id, exposed_items, user_feedback, user_seqfeature, \
            user_protrait, item_feature, _ = FeatureUtil.record_split(record)
            role_id, dense_feature, slate_label, label = 0, [0], [0] * 9, 0
            return [
                role_id,
                [user_seqfeature, [0]],
                user_protrait[10:],
                user_protrait[:10],
                slate_label,
                label
            ]

        state = list(map(fn, records))
        return state

    def get_location_mask(self, location_mask, cur_layer):
        location_mask = location_mask[cur_layer][np.newaxis, :]
        location_mask = np.repeat(location_mask, self.batch_size, 0)
        return location_mask

    @property
    def state(self):
        if self.config.get("support_rllib_mask", False):
            location_mask = self.get_location_mask(self.location_mask, self.cur_steps // 3)
            return {
                "state": self._state,
                "action_mask": self.action_mask & location_mask & self.special_mask
            }
        elif self.config.get("support_d3rl_mask", False):
            cur_steps = np.full((self.batch_size, 1), self.cur_steps)
            return {
                "state": self._state,
                "masked_actions": self.prev_actions,
                "cur_steps": cur_steps
            }
        else:
            return self._state

    @property
    def user(self):
        return [x.split('@')[1] for x in self.records]

    def get_price(self, actions):
        actions_shape = actions.shape
        prices = [self.item_info_d[str(y)]["price"] for y in actions.flatten()]
        return np.array(prices).reshape(actions_shape)

    def get_complete_states(self):
        states = []
        for j in range(self.max_steps):
            tmp = copy(self._init_state)
            for state, action, i in zip(self._init_state, self.prev_actions[:, j], range(len(self._init_state))):
                # dense
                prev_item_feat = [self.item_info_d[str(x)]['item_vec'] for x in self.prev_actions[i, :]]
                cur_item_feat = self.item_info_d[str(action)]['item_vec']
                prev_item_feat = np.array(prev_item_feat).flatten()
                tmp[i][2] = np.concatenate((tmp[i][2], prev_item_feat, cur_item_feat))
                # category
                sequence_id = 1
                tmp[i][3] = np.concatenate((tmp[i][3], [sequence_id], self.prev_actions[i, :], [action]))
            states.append(tmp)
        return states

    def get_violation(self):
        tmp = np.ones((self.batch_size,), dtype=np.int)
        for step in range(self.cur_steps):
            location_mask = self.location_mask[step // 3]
            tmp = tmp & location_mask[self.prev_actions[:, step]]
        for step in range(max(self.cur_steps - 1, 1)):
            duplicate_mask = (self.prev_actions[:, step] != self.prev_actions[:, step + 1])
            tmp = tmp & duplicate_mask
        for step in range(max(self.cur_steps - 2, 1)):
            duplicate_mask = (self.prev_actions[:, step] != self.prev_actions[:, step + 2])
            tmp = tmp & duplicate_mask
        for i in range(self.batch_size):
            if len(np.intersect1d(self.prev_actions[i], self.special_items)) > 1:
                tmp[i] = 0
        return tmp

    @property
    def offline_action(self):
        cur_step = self.cur_steps
        if cur_step < self.max_steps:
            if self.config.get("support_conti_env", False):
                action = [self.action_emb[int(x.split('@')[3].split(',')[cur_step])] for x in self.records]
            else:
                action = [int(x.split('@')[3].split(',')[cur_step]) for x in self.records]
        else:
            if self.config.get("support_conti_env", False):
                action = [self.action_emb[0], ] * self.batch_size
            else:
                action = [0, ] * self.batch_size
        return action

    @property
    def offline_reward(self):
        cur_step = self.cur_steps
        if cur_step < self.max_steps:
            reward = [0, ] * self.batch_size
        else:
            action = np.array([list(map(int, x.split('@')[3].split(','))) for x in self.records])
            price = self.get_price(action)
            slate_label = np.array([list(map(int, x.split('@')[4].split(','))) for x in self.records])
            reward = [sum([xx * yy for (xx, yy) in zip(x, y)]) for (x, y) in zip(price, slate_label)]
        return reward

    @property
    def info(self):
        return self.infos

    @staticmethod
    def get_nearest_neighbor(actions, action_emb, temperature=None):
        action_score = np.einsum('ij,kj->ik', np.array(actions), action_emb)
        best_action = np.argmax(action_score, axis=1)
        return best_action

    @staticmethod
    def get_nearest_neighbor_with_mask(actions, action_emb, action_mask, temperature=None):
        action_score = np.einsum('ij,kj->ik', np.array(actions), action_emb)
        action_score[action_mask < 0.5] = -2 ** 31
        best_action = np.argmax(action_score, axis=1)
        return best_action

    def act(self, actions):
        if self.config.get("support_conti_env", False):
            location_mask = self.get_location_mask(self.location_mask, self.cur_steps // 3)
            action_mask = self.action_mask & location_mask & self.special_mask
            actions = self.get_nearest_neighbor_with_mask(actions, self.action_emb, action_mask)
        self.prev_actions[:, self.cur_steps] = actions
        self.action_mask[list(range(self.batch_size)), actions] = 0
        for i in range(self.batch_size):
            if len(np.intersect1d(self.prev_actions[i], self.special_items)) > 0:
                self.special_mask[i][self.special_items] = 0
        tmp = copy(self._init_state)
        for state, action, i in zip(self._state, actions, range(len(actions))):
            # dense
            prev_item_feat = [self.item_info_d[str(x)]['item_vec'] for x in self.prev_actions[i, :]]
            cur_item_feat = self.item_info_d[str(action)]['item_vec']
            prev_item_feat = np.array(prev_item_feat).flatten()
            tmp[i][2] = np.concatenate((tmp[i][2], prev_item_feat, cur_item_feat))
            # category
            sequence_id = 1
            tmp[i][3] = np.concatenate((tmp[i][3], [sequence_id], self.prev_actions[i, :], [action]))
        self._state = tmp
        self.cur_steps += 1

    def to_string(self):
        return '\n'.join(self.records)


class SlateRecEnv(RecSimBase):
    """ Implements core recommendation simulator"""

    def __init__(self, config, state_cls):
        self.max_steps = config['max_steps']
        self.batch_size = config['batch_size']
        self.FeatureUtil = FeatureUtil(config)
        super().__init__(config, state_cls)
        with self.sess.as_default():
            with self.graph.as_default():
                layer = [x.name for x in self.model.layers
                         if 'simulator_reward' in x.name][0]
                self.reward_layer = tf.keras.backend.function(self.model.input,
                                                              self.model.get_layer(layer).output)
                layer = [x.name for x in self.model.layers
                         if 'simulator_obs' in x.name][0]
                self.obs_layer = tf.keras.backend.function(self.model.input,
                                                           self.model.get_layer(layer).output)

    def get_model(self, config):
        model_type = config.get('algo', 'dien')
        model = __import__("rl4rs.nets." + model_type, fromlist=['get_model']).get_model(config)
        return model

    def obs_fn(self, state):
        if self.config.get("support_rllib_mask", False) or \
                self.config.get("support_d3rl_mask", False):
            feat, _ = self.FeatureUtil.feature_extraction(state["state"])
        else:
            feat, _ = self.FeatureUtil.feature_extraction(state)
        if self.config.get("rawstate_as_obs", False):
            obs = [{
                "category_feature": feat[2][i],
                "dense_feature": feat[1][i],
                "sequence_feature": feat[0][i],
            } for i in range(self.batch_size)]
            if self.config.get("support_rllib_mask", False):
                action_mask = state["action_mask"]
                return [{
                    "action_mask": action_mask[i],
                    **obs[i],
                } for i in range(self.batch_size)]
            else:
                return obs
        else:
            with self.sess.as_default():
                with self.sess.graph.as_default():
                    obs = self.obs_layer(feat)
            if self.config.get("support_rllib_mask", False):
                action_mask = state["action_mask"]
                return [{
                    "action_mask": action_mask[i],
                    "obs": obs[i],
                } for i in range(self.batch_size)]
            elif self.config.get("support_d3rl_mask", False):
                masked_actions = state["masked_actions"]
                cur_steps = state["cur_steps"]
                return np.concatenate([obs, masked_actions, cur_steps], axis=-1)
            else:
                return obs

    def forward(self, model, samples):
        step = samples.cur_steps
        if step < self.max_steps:
            return [0] * self.batch_size
        else:
            # state = samples.state
            prev_actions = samples.prev_actions
            shapes = prev_actions.shape
            complete_states = np.array(samples.get_complete_states())
            complete_states = complete_states \
                .swapaxes(0, 1) \
                .reshape((shapes[0] * shapes[1], 6))
            price = samples.get_price(prev_actions)
            feat, _ = self.FeatureUtil.feature_extraction(complete_states)
            with self.sess.as_default():
                with self.graph.as_default():
                    res = self.reward_layer(feat)
            probs = np.array(res)[:, 1].reshape(shapes)
            if self.config.get("simulator_info_fetch", False):
                [samples.info[i].update({'click_p': probs[i]})
                 for i in range(len(probs))]
            reward = np.sum(price * probs, axis=1)
            if 1:
            # if self.config.get("support_rllib_mask", False) or \
            #         self.config.get("support_d3rl_mask", False):
                violation = samples.get_violation()
                reward[violation < 0.5] = 0
        return reward.tolist()
