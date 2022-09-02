import os
import numpy as np
import gym
import ray
from copy import deepcopy
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState
from rl4rs.utils.rllib_print import pretty_print
from rl4rs.nets.rllib.rllib_rawstate_model import getTFModelWithRawState
from rl4rs.nets.rllib.rllib_mask_model import getMaskActionsModel, \
    getMaskActionsModelWithRawState
from rl4rs.utils.rllib_vector_env import MyVectorEnvWrapper
from script.modelfree_trainer import get_rl_model
from rl4rs.policy.behavior_model import behavior_model
from script.offline_evaluation import ope_eval
from rl4rs.utils.fileutil import find_newest_files
import http.client

http.client.HTTPConnection._http_vsn = 10
http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

import sys

algo = sys.argv[1]
stage = sys.argv[2]
extra_config = eval(sys.argv[3]) if len(sys.argv) >= 4 else {}

ray.init()

config = {"epoch": 10000, "maxlen": 64, "batch_size": 64, "action_size": 284, "class_num": 2, "dense_feature_num": 432,
          "category_feature_num": 21, "category_hash_size": 100000, "seq_num": 2, "emb_size": 128, "is_eval": False,
          "hidden_units": 128, "max_steps": 9, "action_emb_size": 32,
          "sample_file": '../output/rl4rs_dataset_a_shuf.csv', "model_file": "../output/rl4rs_dataset_a_dnn/model",
          "iteminfo_file": '../dataset/item_info.csv', "support_rllib_mask": True,
          "remote_base": 'http://127.0.0.1:16773', 'env': "SlateRecEnv-v0"}

config = dict(config, **extra_config)
eval_config = deepcopy(config)

if config['env'] == 'SeqSlateRecEnv-v0':
    eval_config['max_steps'] = config['max_steps'] = 36
    eval_config['batch_size'] = config['batch_size'] = config['batch_size'] // 4

if "DDPG" in algo or "TD3" in algo or 'conti' in algo:
    eval_config['support_conti_env'] = config['support_conti_env'] = True
    eval_config['support_rllib_mask'] = config['support_rllib_mask'] = False

if 'RAINBOW' in algo:
    eval_config['support_rllib_mask'] = config['support_rllib_mask'] = False
    eval_config['support_onehot_action'] = True
    eval_config['support_conti_env'] = True

if 'rawstate' in algo:
    eval_config['rawstate_as_obs'] = config['rawstate_as_obs'] = True


print(extra_config, config)

mask_model = getMaskActionsModel(true_obs_shape=(256,), action_size=config['action_size'])
ModelCatalog.register_custom_model("mask_model", mask_model)
mask_model_rawstate = getMaskActionsModelWithRawState(config=config, action_size=config['action_size'])
ModelCatalog.register_custom_model("mask_model_rawstate", mask_model_rawstate)
model_rawstate = getTFModelWithRawState(config=config)
ModelCatalog.register_custom_model("model_rawstate", model_rawstate)
register_env('rllibEnv-v0', lambda _: MyVectorEnvWrapper(gym.make('HttpEnv-v0', env_id=config['env'], config=config), config['batch_size']))

modelfile = algo + '_' + config['env'] + '_' + config['trial_name']
output_dir = os.environ['rl4rs_output_dir']
checkpoint_dir = '%s/ray_results/%s/' % (output_dir, modelfile)
restore_dir = find_newest_files('checkpoint*', checkpoint_dir)
restore_file = find_newest_files('checkpoint*', restore_dir)
restore_file = restore_file[:restore_file.rfind('.')] \
    if '.' in restore_file.split('/')[-1] \
    else restore_file
# algo = "DQN"
# algo = "PPO"
if algo == "DDPG" or algo == "DDPG_rawstate":
    assert config['support_conti_env'] == True
    cfg = {
        "exploration_config": {
            "type": "OrnsteinUhlenbeckNoise",
        },
    }
    if 'rawstate' in algo or config.get('rawstate_as_obs', False):
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "model_rawstate",
            }})
if algo == "TD3" or algo == "TD3_rawstate":
    assert config['support_conti_env'] == True
    cfg = {
        "exploration_config": {
            "type": "OrnsteinUhlenbeckNoise",
            "random_timesteps":10000
        },
    }
    if 'rawstate' in algo or config.get('rawstate_as_obs', False):
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "model_rawstate",
            }})
elif algo == "DQN" or algo == "DQN_rawstate":
    cfg = {
        # TODO(ekl) we need to set these to prevent the masked values
        # from being further processed in DistributionalQModel, which
        # would mess up the masking. It is possible to support these if we
        # defined a custom DistributionalQModel that is aware of masking.
        "hiddens": [],
        "dueling": False,
        # Whether to use double dqn
        "double_q": True,
        # N-step Q learning
        "n_step": 1,
        "target_network_update_freq": 200,
        # === Replay buffer ===
        # Size of the replay buffer in batches (not timesteps!).
        "buffer_size": 100000,
        # 'rollout_fragment_length': 200,
        # "num_workers": 0,
        "model": {
            "custom_model": "mask_model",
        },
    }
    if 'rawstate' in algo or config.get('rawstate_as_obs', False):
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
elif algo == "SLATEQ" or algo == "SLATEQ_rawstate":
    cfg = {
        "model": {
            "custom_model": "mask_model",
        },
    }
    if 'rawstate' in algo or config.get('rawstate_as_obs', False):
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
elif algo == "RAINBOW" or algo == "RAINBOW_rawstate":
    # note that DistributionalQModel will make action masking not work
    cfg = {
        # TODO(ekl) we need to set these to prevent the masked values
        # from being further processed in DistributionalQModel, which
        # would mess up the masking. It is possible to support these if we
        # defined a custom DistributionalQModel that is aware of masking.
        "hiddens": [128],
        "noisy": False,
        "num_atoms":8,
        # "dueling": True,
        # Whether to use double dqn
        # "double_q": True,
        # N-step Q learning
        "n_step": 3,
        # "target_network_update_freq": 200,
        "v_min": 0.0,
        "v_max": 1000.0,
        # === Replay buffer ===
        # Size of the replay buffer in batches (not timesteps!).
        "buffer_size": 100000,
        # 'rollout_fragment_length': 200,
        # "num_workers": 0,
        # "model": {
        #     "custom_model": "mask_model",
        # },
    }
    if 'rawstate' in algo or config.get('rawstate_as_obs', False):
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
elif "PPO" in algo:
    cfg = {
        "num_workers": 2,
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # The GAE (lambda) parameter.
        "lambda": 1.0,
        # Initial coefficient for KL divergence.
        "kl_coeff": 0.2,
        # # Size of batches collected from each worker.
        # "rollout_fragment_length": 256,
        # # Number of timesteps collected for each SGD round. This defines the size
        # # of each SGD epoch.
        # "train_batch_size": 2048,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 256,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 1,
        # Stepsize of SGD.
        "lr": 0.0001,
        # Coefficient of the value function loss. IMPORTANT: you must tune this if
        # you set vf_share_layers=True inside your model's config.
        "vf_loss_coeff": 0.5,
        # PPO clip parameter.
        "clip_param": 0.3,
        # Clip param for the value function. Note that this is sensitive to the
        # scale of the rewards. If your expected V is large, increase this.
        "vf_clip_param": 500.0,
        # If specified, clip the global norm of gradients by this amount.
        # "grad_clip": 10.0,
        # Target value for KL divergence.
        "kl_target": 0.01,
    }
    is_rawstate = 'rawstate' in algo or config.get('rawstate_as_obs', False)
    is_conti = 'conti' in algo or config.get('support_conti_env', False)
    if is_conti:
        assert config['support_conti_env'] == True
        cfg = dict({
            **cfg,
            "exploration_config": {
                "type": "StochasticSampling",
            }})
    if is_rawstate and is_conti:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "model_rawstate",
            }})
    elif is_conti:
        pass
    elif is_rawstate:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
    else:
        cfg = dict({
            **cfg,
            "model": {
                "vf_share_layers": False,
                "custom_model": "mask_model",
            }})
elif "A2C" in algo:
    cfg = {
        # Should use a critic as a baseline (otherwise don't use value baseline;
        # required for using GAE).
        "use_critic": True,
        # If true, use the Generalized Advantage Estimator (GAE)
        # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
        "use_gae": True,
        # GAE(gamma) parameter
        "lambda": 1.0,
        # Max global norm for each gradient calculated by worker
        "grad_clip": 10.0,
        # Learning rate
        "lr": 0.0001,
        # Value Function Loss coefficient
        "vf_loss_coeff": 0.5,
        # Entropy coefficient
        "entropy_coeff": 0.01,
        # Min time per iteration
        "min_iter_time_s": 5,
        # "num_workers": 0,
    }
    is_rawstate = 'rawstate' in algo or config.get('rawstate_as_obs', False)
    is_conti = 'conti' in algo or config.get('support_conti_env', False)
    if is_conti:
        assert config['support_conti_env'] == True
        cfg = dict({
            **cfg,
            "exploration_config": {
                "type": "StochasticSampling",
            }})
    if is_rawstate and is_conti:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "model_rawstate",
            }})
    elif is_conti:
        pass
    elif is_rawstate:
        cfg = dict({
            **cfg,
            "use_gae": False,
            "exploration_config": {
                "type": "EpsilonGreedy",
                "final_epsilon": 0.1,
                "epsilon_timesteps": 100000,
            },
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
    else:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model",
            }})

elif "PG" in algo:
    cfg = {
        # "num_workers": 0,
        "lr": 0.0004,
        # "exploration_config": {
        #     "type": "EpsilonGreedy",
        #     "final_epsilon": 0.15,
        # }
    }
    is_rawstate = 'rawstate' in algo or config.get('rawstate_as_obs', False)
    is_conti = 'conti' in algo or config.get('support_conti_env', False)
    if is_conti:
        assert config['support_conti_env'] == True
        cfg = dict({
            **cfg,
            "exploration_config": {
                "type": "StochasticSampling",
            }})
    if is_rawstate and is_conti:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "model_rawstate",
            }})
    elif is_conti:
        pass
    elif is_rawstate:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
    else:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model",
            }})

elif "IMPALA" in algo:
    cfg = {
        # "rollout_fragment_length": 9,
        "min_iter_time_s": 10,
        "num_workers": 2,
        # Learning params.
        "grad_clip": 10.0,
        # Either "adam" or "rmsprop".
        "opt_type": "adam",
        "lr": 0.0001,
        # Balancing the three losses.
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "batch_mode": "truncate_episodes",
        # "_separate_vf_optimizer": True,
        # "_lr_vf": 0.0001,
    }
    is_rawstate = 'rawstate' in algo or config.get('rawstate_as_obs', False)
    is_conti = 'conti' in algo or config.get('support_conti_env', False)
    if is_conti:
        assert config['support_conti_env'] == True
        cfg = dict({
            **cfg,
            "exploration_config": {
                "type": "StochasticSampling",
            }})
    if is_rawstate and is_conti:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "model_rawstate",
            }})
    elif is_conti:
        pass
    elif is_rawstate:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model_rawstate",
            }})
    else:
        cfg = dict({
            **cfg,
            "model": {
                "custom_model": "mask_model",
            }})
else:
    raise Exception

rllib_config = dict(
    {
        "env": "rllibEnv-v0",
        "gamma": 1,
        "explore": True,
        "exploration_config": {
            "type": "SoftQ",
            # "temperature": 1.0,
        },
        "num_gpus": 1 if config.get('gpu', True) else 0,
        "num_workers": 0,
        "framework": 'tf' if 'SLATEQ' not in algo else 'torch',
        # "framework": 'tfe',
        "rollout_fragment_length": config['max_steps'],
        "batch_mode": "complete_episodes",
        "train_batch_size": min(config["batch_size"] * config['max_steps'], 1024),
        "evaluation_interval": 500,
        "evaluation_num_episodes": 2048 * 4,
        "evaluation_config": {
            "explore": False
        },
        "log_level": "INFO",
    },
    **cfg)
print('rllib_config', rllib_config)
trainer = get_rl_model(algo.split('_')[0], rllib_config)

if stage == 'train':
    try:
        trainer.restore(restore_file)
        print('model restore from %s' % (restore_file))
    except Exception:
        trainer = get_rl_model(algo.split('_')[0], rllib_config)
    for i in range(config["epoch"]):
        result = trainer.train()
        if (i + 1) % 50 == 0:
            print('epoch ',i)
        if (i + 1) % 500 == 0 or i == 0:
            print(pretty_print(result))
        if (i + 1) % 500 == 0:
            checkpoint = trainer.save(checkpoint_dir=checkpoint_dir)
            print("checkpoint saved at", checkpoint)

if stage == 'eval':
    eval_config = config.copy()
    eval_config['is_eval'] = True
    eval_config['batch_size'] = 2048
    eval_env = gym.make('HttpEnv-v0', env_id=eval_config['env'], config=eval_config)
    # trainer.restore(checkpoint_dir + '/checkpoint_010000/checkpoint-10000')
    trainer.restore(restore_file)
    print('model restore from %s' % (restore_file))
    episode_reward = 0
    done = False
    epoch = 4
    actions = []
    for i in range(epoch):
        obs = eval_env.reset()
        print('test batch at ', i, 'avg reward', episode_reward / eval_config['batch_size'] / (i + 0.0001))
        for _ in range(config["max_steps"]):
            obs = dict(enumerate(obs))
            action = trainer.compute_actions(obs, explore=False)
            action = np.array(list(action.values()))
            obs, reward, done, info = eval_env.step(action)
            episode_reward += sum(reward)
            actions.append(action)
    print('avg reward', episode_reward / eval_config['batch_size'] / epoch)
    eval_env.close()


if stage == 'eval_v2':
    # eval_config["epoch"] = 1
    eval_config['is_eval'] = True
    eval_config["batch_size"] = 2048
    if config['env'] == 'SeqSlateRecEnv-v0':
        config['max_steps'] = 36
        sim = SeqSlateRecEnv(eval_config, state_cls=SeqSlateState)
        eval_env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    else:
        sim = SlateRecEnv(eval_config, state_cls=SlateState)
        eval_env = gym.make('SlateRecEnv-v0', recsim=sim)
    # trainer.restore(checkpoint_dir + '/checkpoint_010000/checkpoint-10000')
    trainer.restore(restore_file)
    print('model restore from %s' % (restore_file))
    from rl4rs.policy.policy_model import policy_model
    policy = policy_model(trainer, config)
    episode_reward = 0
    done = False
    epoch = 4
    actions = []
    for i in range(epoch):
        obs = eval_env.reset()
        print('test batch at ', i, 'avg reward', episode_reward / eval_config['batch_size'] / (i + 0.0001))
        for _ in range(config["max_steps"]):
            if config.get('support_onehot_action', False):
                action = policy.predict_with_mask(obs)
            else:
                action = np.array(policy.action_probs(obs))
            obs, reward, done, info = eval_env.step(action)
            episode_reward += sum(reward)
            actions.append(action)
    print('avg reward', episode_reward / eval_config['batch_size'] / epoch)
    eval_env.close()


if stage == 'ope':
    dataset_dir = os.environ['rl4rs_dataset_dir']
    sample_model = behavior_model(config, modelfile=dataset_dir + '/logged_policy.h5')
    trainer.restore(restore_file)
    print('model restore from %s' % (restore_file))
    eval_config = config.copy()
    eval_config["epoch"] = 1
    eval_config['is_eval'] = True
    eval_config["batch_size"] = 2048
    if config['env'] == 'SeqSlateRecEnv-v0':
        config['max_steps'] = 36
        sim = SeqSlateRecEnv(eval_config, state_cls=SeqSlateState)
        eval_env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    else:
        sim = SlateRecEnv(eval_config, state_cls=SlateState)
        eval_env = gym.make('SlateRecEnv-v0', recsim=sim)
    ope_eval(eval_config, eval_env, trainer, sample_model=sample_model)

ray.shutdown()
