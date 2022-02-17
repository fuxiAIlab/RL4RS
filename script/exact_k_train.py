# -*- coding: utf-8 -*-
import os, sys
import gym
import numpy as np
import tensorflow as tf
from rl4rs.nets.exact_k.model import Generator, Discriminator
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState
from rl4rs.utils.fileutil import find_newest_files

stage = sys.argv[1]
extra_config = eval(sys.argv[2])

config = {"epoch": 10000, "maxlen": 64, "batch_size": 256, "action_size": 284, "class_num": 2, "dense_feature_num": 432,
          "category_feature_num": 21, "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "page_items": 9,
          "hidden_units": 128, "max_steps": 9, "sample_file": '../dataset/rl4rs_dataset_b3_shuf.csv', "iteminfo_file": '../item_info.csv',
          "model_file": "../output/simulator_b2_dien/model", "support_rllib_mask": False, "is_eval": False, 'env': "SlateRecEnv-v0"}

config = dict(config, **extra_config)

if config['env'] == 'SeqSlateRecEnv-v0':
    config['max_steps'] = 36
    sim = SeqSlateRecEnv(config, state_cls=SeqSlateState)
    env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
else:
    sim = SlateRecEnv(config, state_cls=SlateState)
    env = gym.make('SlateRecEnv-v0', recsim=sim)

batch_size = config["batch_size"]
action_size = config["action_size"]
epoch = config["epoch"]
max_steps = config["max_steps"]
output_dir = os.environ['rl4rs_output_dir']
model_dir = '%s/%s/' % (output_dir, 'exactk_' + config['env'] + '_' + config['trial_name'])
model_save_path = model_dir + 'exact_k.ckpt'
restore_file = find_newest_files('exact_k.ckpt*', model_dir)
restore_file = restore_file[:restore_file.rfind('.')]

l0_ssr_mask = np.zeros(action_size)
location_mask, special_items = SlateState.get_mask_from_file(config['iteminfo_file'], action_size)
l1_mask, l2_mask, l3_mask = location_mask[0], location_mask[1], location_mask[2]
l0_ssr_mask[special_items] = 1

with tf.name_scope('Generator'):
    g = Generator(l1_mask,
                  l2_mask,
                  l3_mask,
                  l0_ssr_mask,
                  is_training=True,
                  seq_length=action_size)

with tf.name_scope('Discriminator'):
    d = Discriminator(seq_length=action_size)

print("Graph loaded")

if config.get('gpu', True):
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=0.5,
        allow_growth=True)  # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    sess_config = tf.ConfigProto()

if stage == 'train':
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.initialize_all_variables())
        print('Generator training start!')
        reward_total = 0.0
        for episode in range(epoch):
            print('Generator episode: ', episode)

            observation = np.array(env.reset())
            item_cand = np.array([list(range(0, config['action_size']))] * batch_size)
            hill_b_f = []
            for i in range(2):
                # get action
                sampled_card_idx, sampled_card = sess.run([g.sampled_path, g.sampled_result],
                                                          feed_dict={g.user: observation, g.item_cand: item_cand})
                for step in range(config['max_steps']):
                    observation_, reward, done, info = env.step(sampled_card[:, step])

                env.reset()
                # hill b f
                hill_b_f.append(list(zip(sampled_card, sampled_card_idx, reward)))

            b_hill_f = np.transpose(hill_b_f, [1, 0, 2])
            samples = []
            for hill_f in b_hill_f:
                sorted_list = sorted(hill_f, key=lambda x: x[2], reverse=True)
                samples.append(sorted_list[np.random.choice(1)])

            (sampled_card, sampled_card_idx, reward) = zip(*samples)
            reward = np.array(reward)

            reward_ = sess.run(d.reward, feed_dict={d.user: observation})
            sess.run(d.train_op, feed_dict={d.user: observation, d.reward_target: reward})

            if episode % 50 == 0:
                print('episode:', episode)
                print('reward_target', np.mean(reward_))
                print('reward', np.mean(reward))
                print('actions', sampled_card[:10])
            reward = (reward - reward_)

            reward = reward / np.std(reward)

            sess.run(g.train_op, feed_dict={g.decode_target_ids: sampled_card_idx,
                                            g.reward: reward,
                                            g.item_cand: item_cand,
                                            g.user: observation,
                                            })
            gs_gen = sess.run(g.global_step)

            if episode % 500 == 0:
                saver = tf.train.Saver()
                saver.save(sess, model_save_path + '.' + str(episode))
                print('save model:' + model_save_path + '.' + str(episode))
        print('Generator training done!')
    saver = tf.train.Saver()
    saver.save(sess, model_save_path + '.' + str(episode))
    print('save model:' + model_save_path + '.' + str(episode))
    print("Done")

if stage == 'eval':
    with tf.Session(config=sess_config) as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver()
        saver.restore(sess, restore_file)
        print('restore exact-k model from %s' % (restore_file))
        episode_reward = 0
        done = False
        epoch = 4
        for i in range(epoch):
            observation = np.array(env.reset())
            item_cand = np.array([list(range(0, config['action_size']))] * batch_size)
            sampled_card_idx, sampled_card = sess.run([g.greedy_path, g.greedy_result],
                                                      feed_dict={g.user: observation, g.item_cand: item_cand})
            for step in range(config['max_steps']):
                observation_, reward, done, info = env.step(sampled_card[:, step])
                episode_reward += sum(reward)
            print('actions', sampled_card[:10])
            print('avg reward', episode_reward / config['batch_size'] / (i + 1))
