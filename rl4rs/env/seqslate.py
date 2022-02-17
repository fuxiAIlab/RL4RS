from functools import reduce
from operator import add
from copy import deepcopy as copy
import numpy as np
from rl4rs.env.slate import SlateState, SlateRecEnv


class SeqSlateState(SlateState):
    def __init__(self, config, records):
        super().__init__(config, records)
        self.page_items = config.get("page_items", 9)

    @property
    def state(self):
        if self.config.get("support_rllib_mask", False):
            location_mask = self.get_location_mask(self.location_mask, self.cur_steps % self.page_items // 3)
            return {"state": self._state, "action_mask": self.action_mask & location_mask & self.special_mask}
        elif self.config.get("support_d3rl_mask", False):
            cur_steps = np.full((self.batch_size, 1), self.cur_steps)
            page_init = self.cur_steps // self.page_items * self.page_items
            page_end = min(page_init + self.page_items - 1, self.max_steps - 1)
            masked_actions = self.prev_actions[:, page_end + 1 - self.page_items:page_end + 1]
            return {"state": self._state, "masked_actions": masked_actions, "cur_steps": cur_steps}
        else:
            return self._state

    def get_complete_states(self):
        states = []
        for j in range(self.cur_steps):
            tmp = copy(self._init_state)
            for state, action, i in zip(self._init_state, self.prev_actions[:, j], range(len(self._init_state))):
                page_init = j // self.page_items * self.page_items
                page_end = page_init + self.page_items - 1
                sequence_id = j // self.page_items + 1
                # seq
                prev_expose = self.prev_actions[i, :page_init] if page_init > 0 else [0]
                tmp[i][1] = [tmp[i][1][0], prev_expose]
                # dense
                prev_item_feat = [
                    self.item_info_d[str(x)]['item_vec']
                    for x in self.prev_actions[i, page_init:page_end + 1]
                ]
                cur_item_feat = self.item_info_d[str(action)]['item_vec']
                prev_item_feat = np.array(prev_item_feat).flatten()
                tmp[i][2] = np.concatenate((tmp[i][2], prev_item_feat, cur_item_feat))
                # category
                cur_exposed = self.prev_actions[i, page_init:page_end + 1]
                tmp[i][3] = np.concatenate((tmp[i][3], [sequence_id], cur_exposed, [action]))
            states.append(tmp)
        return states

    def get_violation(self):
        tmp = np.ones((self.batch_size,), dtype=np.int)
        for step in range(self.cur_steps):
            location_mask = self.location_mask[step % self.page_items // 3]
            tmp = tmp & location_mask[self.prev_actions[:, step]]
        for step in range(max(self.cur_steps - 1, 1)):
            duplicate_mask = (self.prev_actions[:, step] != self.prev_actions[:, step + 1])
            tmp = tmp & duplicate_mask
        for step in range(max(self.cur_steps - 2, 1)):
            duplicate_mask = (self.prev_actions[:, step] != self.prev_actions[:, step + 2])
            tmp = tmp & duplicate_mask
        for i in range(self.batch_size):
            cur_page = self.cur_steps % self.page_items
            for j in range(cur_page+1):
                actions = self.prev_actions[i][self.page_items*j:self.page_items*(j+1)]
                if len(np.intersect1d(actions, self.special_items)) > 1:
                    tmp[i] = 0
        return tmp

    @property
    def offline_reward(self):
        cur_step = self.cur_steps
        if cur_step % 9 != 0:
            reward = [0, ] * self.batch_size
        else:
            action = np.array([list(map(int, x.split('@')[3].split(',')[:cur_step]))
                               for x in self.records])
            price = self.get_price(action)[:, -self.page_items:]
            slate_label = np.array([
                list(map(int, x.split('@')[4].split(',')))
                for x in self.records
            ])
            slate_label = slate_label[:, cur_step - self.page_items:cur_step]
            reward = np.sum(price * slate_label, axis=1)
        return reward

    # @property
    # def info(self):
    #     return [{}]*self.batch_size

    def act(self, actions):
        if self.config.get("support_conti_env", False):
            location_mask = self.get_location_mask(self.location_mask,
                                                   self.cur_steps % self.page_items // 3)
            action_mask = self.action_mask & location_mask & self.special_mask
            actions = self.get_nearest_neighbor_with_mask(actions, self.action_emb, action_mask)
        self.prev_actions[:, self.cur_steps] = actions
        self.action_mask[list(range(self.batch_size)), actions] = 0
        for i in range(self.batch_size):
            if len(np.intersect1d(self.prev_actions[i], self.special_items)) > 0:
                self.special_mask[i][self.special_items] = 0
        tmp = copy(self._init_state)
        for state, action, i in zip(self._state, actions, range(self.batch_size)):
            page_init = self.cur_steps // self.page_items * self.page_items
            page_end = page_init + self.page_items - 1
            sequence_id = self.cur_steps // self.page_items + 1
            # seq
            prev_expose = self.prev_actions[i, :page_init] if page_init > 0 else [0]
            tmp[i][1] = [tmp[i][1][0], prev_expose]
            # dense
            prev_item_feat = [
                self.item_info_d[str(x)]['item_vec']
                for x in self.prev_actions[i, page_init:page_end + 1]
            ]
            cur_item_feat = self.item_info_d[str(action)]['item_vec']
            prev_item_feat = np.array(prev_item_feat).flatten()
            tmp[i][2] = np.concatenate((tmp[i][2], prev_item_feat, cur_item_feat))
            # category
            cur_exposed = self.prev_actions[i, page_init:page_end + 1]
            tmp[i][3] = np.concatenate((tmp[i][3], [sequence_id], cur_exposed, [action]))
        self._state = tmp
        self.cur_steps += 1
        if self.cur_steps % self.page_items == 0:
            self.action_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)
            self.special_mask = np.full((self.batch_size, self.action_size), 1, dtype=np.int)


class SeqSlateRecEnv(SlateRecEnv):
    """ Implements core recommendation simulator"""

    def __init__(self, config, state_cls):
        super().__init__(config, state_cls)
        self.page_items = config.get("page_items", 9)

    def forward(self, model, samples):
        step = samples.cur_steps
        if step % self.page_items == 0:
            # state = samples.state
            prev_actions = samples.prev_actions[:, :step]
            # shapes = prev_actions.shape
            complete_states = np.array(samples.get_complete_states())
            complete_states = complete_states[-self.page_items:]
            complete_states = complete_states \
                .swapaxes(0, 1) \
                .reshape((self.batch_size * self.page_items, 6))
            price = samples.get_price(prev_actions)[:, -self.page_items:]
            feat, _ = self.FeatureUtil.feature_extraction(complete_states)
            with self.sess.as_default():
                with self.graph.as_default():
                    res = self.reward_layer(feat)
            probs = np.array(res)[:, 1].reshape((self.batch_size, self.page_items))
            reward = np.sum(price * probs, axis=1)
            if self.config.get("support_rllib_mask", False) or \
                    self.config.get("support_d3rl_mask", False):
                violation = samples.get_violation()
                reward[violation < 0.5] = 0
        else:
            reward = np.array([0, ] * self.batch_size)
        return reward.tolist()
