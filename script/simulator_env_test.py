import os, sys
import gym
import numpy as np
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from rl4rs.utils.datautil import FeatureUtil
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState

extra_config = eval(sys.argv[1]) if len(sys.argv) >= 2 else {}

config = {"epoch": 1, "maxlen": 64, "batch_size": 2048, "action_size": 284, "class_num": 2, "dense_feature_num": 432,
          "category_feature_num": 21, "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "page_items": 9,
          "hidden_units": 128, "max_steps": 9, "sample_file": '../dataset/rl4rs_dataset_a_train.csv',
          "iteminfo_file": '../dataset/item_info.csv', "tfrecord_file":'../output/rl4rs_dataset_a_train_tiny.tfrecord',
          "model_file": "../output/supervised_a_train_dien/model", "support_rllib_mask": False, "is_eval": True, 'env': "SlateRecEnv-v0",
          "support_conti_env":True, "rawstate_as_obs":False}

config = dict(config, **extra_config)

if config.get('gpu', 0) < 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if config['env'] == 'SeqSlateRecEnv-v0':
    config['max_steps'] = 36
    sim = SeqSlateRecEnv(config, state_cls=SeqSlateState)
    env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
else:
    sim = SlateRecEnv(config, state_cls=SlateState)
    env = gym.make('SlateRecEnv-v0', recsim=sim)

batch_size = config["batch_size"]
epoch = config["epoch"]
max_steps = config["max_steps"]
rewards = np.zeros((epoch, batch_size))
offline_rewards = np.zeros((epoch, batch_size))
offline_actions = np.zeros((epoch, batch_size, max_steps))
for i in range(epoch):
    env.reset(reset_file=True)
    for j in range(config["max_steps"]):
        if not config.get("support_conti_env"):
            action = env.offline_action
        else:
            action = np.full((batch_size, 32), 1)
        offline_actions[i, :, j] = env.offline_action
        next_obs, reward, done, info = env.step(action)
        rewards[i] = rewards[i] + np.array(reward)
        offline_rewards[i] = offline_rewards[i] + np.array(env.offline_reward)
        if done[0]:
            print(next_obs[0], reward[0], action[0], done[0], info[0])
            break

    if config['rawstate_as_obs']:
        config['batch_size'] = 1
        featureutil = FeatureUtil(config)
        iter_train = featureutil.read_tfrecord(config['tfrecord_file'], is_slate_label=False)
        feature = iter_train.make_one_shot_iterator().get_next()
        seq_feature = feature[0][0].numpy()[0]
        dense_feature = feature[0][1].numpy()[0]
        category_feature = feature[0][2].numpy()[0]
        assert np.min(np.equal(next_obs[0]['category_feature'][:-1], category_feature[:-1]))
        assert np.min(np.equal(next_obs[0]['dense_feature'][:-40], dense_feature[:-40]))
        assert np.min(np.equal(next_obs[0]['sequence_feature'], seq_feature))
