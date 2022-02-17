import os
import gym
import numpy as np
from rl4rs.policy.behavior_model import behavior_model
from rl4rs.policy.policy_model import policy_model
import rl4rs.utils.offline_policy_metrics as OPE


def ope_eval(config, eval_env, algo, sample_model: behavior_model = None):
    policy = policy_model(algo, config)
    metrics = []
    epoch = config["epoch"]
    batch_size = config["batch_size"]
    max_steps = config["max_steps"]
    page_items = config.get("page_items", 9)
    for i in range(epoch):
        obs = eval_env.reset()
        episode_rewards, q_values, off_rewards = [], [], []
        prev_actions = []
        action_probs, behavior_probs, rewards = [], [], []
        print('test batch at ', i)
        for j in range(max_steps):
            # obs = dict(enumerate(obs))
            action = policy.predict_with_mask(obs)
            off_action = eval_env.offline_action
            if sample_model is not None:
                action_prob = policy.action_probs(obs)
                action_prob = action_prob[range(batch_size), off_action]
                q_values.append(policy.predict_q(obs, action))
                action_probs.append(action_prob)
                behavior_prob = sample_model.action_probs(eval_env.samples.records, off_action, j // 3 + 1, page=j//page_items)
                behavior_probs.append(behavior_prob)
            obs, reward, done, info = eval_env.step(action)
            off_rewards.append(eval_env.offline_reward)
            rewards.append(reward)
            prev_actions.append(action)

        episode_reward = np.sum(np.array(rewards), axis=0)
        episode_rewards.append(episode_reward)
        if sample_model is not None:
            action_probs = np.array(action_probs).swapaxes(0, 1)
            behavior_probs = np.array(behavior_probs).swapaxes(0, 1)
            off_rewards = np.array(off_rewards).swapaxes(0, 1)
            off_rewards_sum = np.sum(off_rewards, axis=1)
            rewards_hat = np.array(rewards).swapaxes(0, 1)
            q_values = np.array(q_values).swapaxes(0, 1)
            # multiply probs
            action_probs_mul = np.multiply.reduce(action_probs*100, axis=1)
            behavior_probs_mul = np.multiply.reduce(behavior_probs*100, axis=1)
            cips = OPE.eval_CIPS(off_rewards_sum, action_probs_mul, behavior_probs_mul)
            # snips = OPE.eval_SNIPS(off_rewards_sum, action_probs_mul, behavior_probs_mul)
            dr = OPE.eval_doubly_robust(
                episode_reward,
                np.average(q_values, 1),
                off_rewards_sum,
                action_probs_mul,
                behavior_probs_mul
            )
            # step-wise
            wips = OPE.eval_WIPS(off_rewards, action_probs, behavior_probs)
            sdr = OPE.eval_seq_doubly_robust(
                rewards_hat,
                q_values,
                off_rewards,
                action_probs,
                behavior_probs
            )

            metrics.append((cips, dr, wips, sdr))

    print('IS', 'DR', 'WIPS', 'SeqDR', sep=' ')
    print(np.average(np.array(metrics), axis=0))
    print(np.std(np.array(metrics), axis=0))
