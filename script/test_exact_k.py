import numpy as np
import tensorflow as tf
from rl4rs.nets.exact_k.model import Generator, Discriminator

batch_size = 2
l1_mask = np.zeros(284)
l1_mask[:40] = 1
l2_mask = np.zeros(284)
l2_mask[40:150] = 1
l3_mask = np.zeros(284)
l3_mask[150:] = 1
l0_ssr_mask = np.zeros(284)
l0_ssr_mask[:30] = 1
l0_ssr_mask[40:140] = 1
l0_ssr_mask[160:] = 1

with tf.name_scope('Generator'):
    g = Generator(l1_mask,
                  l2_mask,
                  l3_mask,
                  l0_ssr_mask,
                  is_training=True,
                  seq_length=284)

with tf.name_scope('Discriminator'):
    d = Discriminator(seq_length=284)

print("Graph loaded")

gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.95,
    allow_growth=True)
sess_config = tf.ConfigProto(allow_soft_placement=True,
                             gpu_options=gpu_options)

with tf.Session(config=sess_config) as sess:
    sess.run(tf.initialize_all_variables())
    print('Generator training start!')

    reward_total = 0.0
    observation = np.random.random((batch_size, 256))
    item_cand = np.array([list(range(0, 284))] * batch_size)
    for _ in range(9):
        sampled_card_idx, sampled_card = sess.run([g.sampled_path, g.sampled_result],
                                                  feed_dict={g.user: observation, g.item_cand: item_cand})
        reward = np.ones((batch_size,))

        reward_ = sess.run(d.reward, feed_dict={d.user: observation})
        sess.run(d.train_op, feed_dict={d.user: observation, d.reward_target: reward})

        reward_total += np.mean(reward)

        reward = (reward - reward_)

        sess.run(g.train_op, feed_dict={g.decode_target_ids: sampled_card_idx,
                                        g.reward: reward,
                                        g.item_cand: item_cand,
                                        g.user: observation,
                                        })
        gs_gen = sess.run(g.global_step)

        # beamsearch
        # beam_card = sess.run(g.infer_result,
        #                      feed_dict={g.item_cand: item_cand,
        #                                 g.enc_user: observation})

        print(sampled_card_idx, sampled_card, reward_)

print("Done")
