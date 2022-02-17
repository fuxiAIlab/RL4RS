import numpy as np
import tensorflow as tf
# tf.compat.v1.enable_eager_execution()
from tensorflow import keras
from rl4rs.utils.datautil import FeatureUtil
from copy import deepcopy


class behavior_model(object):
    def __init__(self, config, modelfile):
        behavior_config = deepcopy(config)
        behavior_config['category_feature_num'] = 21
        behavior_config['dense_feature_num'] = 50
        self.featureutil = FeatureUtil(behavior_config)
        self.item_feature_size = config.get('item_feature_size', 40)
        self.page_items = config.get("page_items", 9)
        self.sess = tf.Session()
        with self.sess.as_default():
            self.model = keras.models.load_model(modelfile)

    def record2input(self, records, page=0):
        inputs = []
        for record in records:
            role_id, _, sequence_id, exposed_items, user_feedback, user_seqfeature, \
                user_protrait, item_feature, _ = self.featureutil.record_split(record)
            category_feature = user_protrait[:10] + \
                               [sequence_id] + \
                               exposed_items[self.page_items*page:self.page_items*(page+1)]
            sequence_feature = [user_seqfeature, [0]]
            label = 0
            dense_feature_size = self.item_feature_size*self.page_items
            item_feature = item_feature[dense_feature_size*page:dense_feature_size*(page+1)]
            item_feature = np.array(item_feature).reshape((self.page_items, self.item_feature_size))
            item_feature = item_feature[:, :5].reshape(-1)
            inputs.append((
                role_id,
                sequence_feature,
                item_feature,
                category_feature,
                user_feedback[self.page_items*page:self.page_items*(page+1)],
                label))
        return inputs

    def action_probs(self, record, action, layer, page=0):
        batch_size = len(action)
        seq, dense, category, slate = self.featureutil.feature_extraction(self.record2input(record, page))[0]
        with self.sess.as_default():
            y = self.model.predict([seq, dense, category, slate])
        if layer == 1:
            action = np.clip(np.array(action) - 1, 0, 38)
            action_probs = y[:, 1:40] / np.sum(y[:, 1:40], axis=1, keepdims=True)
        elif layer == 2:
            action = np.clip(np.array(action) - 40, 0, 107)
            action_probs = y[:, 40:148] / np.sum(y[:, 40:148], axis=1, keepdims=True)
        else:
            action = np.clip(np.array(action) - 148, 0, 233)
            action_probs = y[:, 148:] / np.sum(y[:, 148:], axis=1, keepdims=True)
        return action_probs[range(batch_size), action]
