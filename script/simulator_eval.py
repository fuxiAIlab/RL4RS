import os, sys
import gym
import numpy as np
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState

extra_config = eval(sys.argv[1]) if len(sys.argv) >= 2 else {}

config = {"epoch": 4, "maxlen": 64, "batch_size": 2048, "action_size": 284, "class_num": 2, "dense_feature_num": 432,
          "category_feature_num": 21, "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "page_items": 9,
          "hidden_units": 128, "max_steps": 9, "sample_file": '../dataset/rl4rs_dataset_b3_shuf.csv', "iteminfo_file": '../item_info.csv',
          "model_file": "../output/simulator_b2_dien/model", "support_rllib_mask": False, "is_eval": True, 'env': "SeqSlateRecEnv-v0"}

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
    env.reset()
    for j in range(config["max_steps"]):
        action = env.offline_action
        offline_actions[i, :, j] = env.offline_action
        next_obs, reward, done, info = env.step(action)
        rewards[i] = rewards[i] + np.array(reward)
        offline_rewards[i] = offline_rewards[i] + np.array(env.offline_reward)
        if done[0]:
            print(
                i,
                np.sum(rewards) / config["batch_size"] / (i + 1),
                np.sum(offline_rewards) / config["batch_size"] / (i + 1)
            )
            break
print('the mean of offline reward', np.mean(offline_rewards))
print('the mean of reward prediction error', np.mean(rewards - offline_rewards))
print('the absolute mean of reward prediction error', np.mean(np.abs(rewards - offline_rewards)))
print('the std of reward prediction error', np.std(np.reshape(rewards - offline_rewards, -1)))
print('success')

