import os
import gym
import d3rlpy
import numpy as np
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState
from d3rlpy.dataset import MDPDataset, Episode
from rl4rs.policy.policy_model import policy_model
from rl4rs.utils import d3rlpy_scorer
from rl4rs.nets.cql.encoder import CustomVectorEncoderFactory
from rl4rs.nets.cql.q_function import CustomMeanQFunctionFactory


def get_model(config, algo, dynamics=None):
    if algo == 'dynamics':
        discrete_action = False if config.get("support_conti_env",False) else True
        encoder_factory = CustomVectorEncoderFactory(
            config,
            action_size=config["action_size"],
            mask_size=int(config['page_items'])+1,
            with_q=False,
            hidden_units=[256, 128],
            use_batch_norm=True,
            dropout_rate = 0.2,
            use_dense=True
        )
        model = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(batch_size=512,
                                                              discrete_action = discrete_action,
                                                              encoder_factory=encoder_factory,
                                                              learning_rate=1e-4,
                                                              scaler = 'min_max',
                                                              reward_scaler = 'standard',
                                                              use_gpu=config['gpu'])
    elif algo == 'BC':
        encoder_factory = CustomVectorEncoderFactory(
            config,
            action_size=config["action_size"],
            mask_size=int(config['page_items'])+1,
            with_q=True,
            hidden_units=[256]
        )
        model = d3rlpy.algos.DiscreteBC(
            batch_size=256,
            beta=0,
            encoder_factory=encoder_factory,
            use_gpu=config['gpu']
        )
    elif algo == 'BCQ':
        encoder_factory = CustomVectorEncoderFactory(
            config,
            action_size=config["action_size"],
            mask_size=int(config['page_items'])+1,
            with_q=True,
            hidden_units=[256]
        )
        model = d3rlpy.algos.DiscreteBCQ(
            batch_size=256,
            encoder_factory=encoder_factory,
            use_gpu=config['gpu']
        )
    elif algo == 'BCQ-conti':
        # encoder_factory = CustomVectorEncoderFactory(
        #     config,
        #     action_size=config["action_size"],
        #     mask_size=int(config['page_items'])+1,
        #     with_q=True,
        #     hidden_units=[256]
        # )
        model = d3rlpy.algos.BCQ(
            batch_size=256,
            # encoder_factory=encoder_factory,
            use_gpu=config['gpu']
        )
    elif algo == 'CQL':
        encoder_factory = CustomVectorEncoderFactory(
            config,
            action_size=config["action_size"],
            mask_size=int(config['page_items'])+1,
            with_q=True,
            hidden_units=[256]
        )
        model = d3rlpy.algos.DiscreteCQL(
            batch_size=256,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
            # encoder_factory=encoder_factory,
            gamma=1.0,
            alpha=config["CQL_alpha"],
            reward_scaler='standard',
            use_gpu=config['gpu']
        )
    elif algo == 'CQL-conti':
        # encoder_factory = CustomVectorEncoderFactory(
        #     config,
        #     action_size=config["action_size"],
        #     mask_size=int(config['page_items'])+1,
        #     with_q=True,
        #     hidden_units=[256]
        # )
        model = d3rlpy.algos.CQL(
            batch_size=256,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
            # encoder_factory=encoder_factory,
            gamma=1.0,
            alpha=config["CQL_alpha"],
            reward_scaler='standard',
            use_gpu=config['gpu']
        )
    elif algo == 'MOPO':
        # encoder_factory = CustomVectorEncoderFactory(
        #     config,
        #     action_size=config["action_size"],
        #     mask_size=int(config['page_items'])+1,
        #     with_q=True,
        #     hidden_units=[256]
        # )
        model = d3rlpy.algos.MOPO(
            dynamics = dynamics,
            batch_size=256,
            update_actor_interval = 2000,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
            # actor_encoder_factory=encoder_factory,
            gamma=1.0,
            scaler = 'min_max',
            reward_scaler='standard',
            use_gpu=config['gpu'],
            # real_ratio = 1,
            # rollout_interval = 1000000000,
            # generated_maxlen = 100
        )
    elif algo == 'COMBO':
        # encoder_factory = CustomVectorEncoderFactory(
        #     config,
        #     action_size=config["action_size"],
        #     mask_size=int(config['page_items'])+1,
        #     with_q=True,
        #     hidden_units=[256]
        # )
        model = d3rlpy.algos.COMBO(
            dynamics = dynamics,
            batch_size=256,
            update_actor_interval = 2000,
            # q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
            # actor_encoder_factory=encoder_factory,
            gamma=1.0,
            scaler = 'min_max',
            reward_scaler='standard',
            use_gpu=config['gpu'],
            # real_ratio = 1,
            # rollout_interval = 1000000000,
            # generated_maxlen = 100
        )
    else:
        assert algo in ('BC', 'BCQ', 'CQL', "MOPO", "COMBO")
    return model


def build_with_dataset(model, dataset):
    model.build_with_dataset(dataset)
    if model._scaler:
        model._scaler.fit(dataset)

        # initialize action scaler
    if model._action_scaler:
        model._action_scaler.fit(dataset)

        # initialize reward scaler
    if model._reward_scaler:
        model._reward_scaler.fit(dataset)
    return model


def data_generate_rl4rs_a(config, datasetfile):
    if config.get('gpu', 0) < 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    batch_size = config["batch_size"]
    sim = SlateRecEnv(config, state_cls=SlateState)
    env = gym.make('SlateRecEnv-v0', recsim=sim)
    epoch = 1000000 // batch_size
    observations = np.zeros((epoch, batch_size, 10, 256 + 9 + 1), 'float32')
    actions = np.zeros((epoch, batch_size, 10), 'float32')
    rewards = np.zeros((epoch, batch_size, 10), 'float32')
    terminals = np.zeros((epoch, batch_size, 10), 'float32')
    for i in range(epoch):
        obs = env.reset()
        observations[i, :, 0, :] = obs
        action = env.offline_action
        actions[i, :, 0] = action
        rewards[i, :, 0] = [0] * batch_size
        terminals[i, :, 0] = [0] * batch_size
        for j in range(9):
            obs, reward, done, info = env.step(action)
            observations[i, :, j + 1, :] = obs
            action = env.offline_action
            actions[i, :, j + 1] = action
            rewards[i, :, j + 1] = env.offline_reward
            terminals[i, :, j + 1] = done

    print('max', np.max(actions))

    # shuffle
    p = np.random.permutation(epoch)
    observations = observations[p]
    actions = actions[p]
    rewards = rewards[p]
    terminals = terminals[p]

    # reshape
    observations.shape = (epoch * batch_size * 10, -1)
    actions.shape = (epoch * batch_size * 10, 1)
    rewards.shape = (epoch * batch_size * 10,)
    terminals.shape = (epoch * batch_size * 10,)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # # save as HDF5
    dataset.dump(datasetfile)


def data_generate_rl4rs_a_conti(config, datasetfile):
    config["support_conti_env"] = 1
    if config.get('gpu', 0) < 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if config.get("support_onehot_action", False):
        config['action_emb_size'] = config["action_size"]

    batch_size = config["batch_size"]
    sim = SlateRecEnv(config, state_cls=SlateState)
    env = gym.make('SlateRecEnv-v0', recsim=sim)
    epoch = 1000000 // batch_size
    observations = np.zeros((epoch, batch_size, 10, 256 + 9 + 1), 'float32')
    actions = np.zeros((epoch, batch_size, 10, config['action_emb_size']), 'float32')
    rewards = np.zeros((epoch, batch_size, 10), 'float32')
    terminals = np.zeros((epoch, batch_size, 10), 'float32')
    for i in range(epoch):
        obs = env.reset()
        observations[i, :, 0, :] = obs
        action = env.offline_action
        actions[i, :, 0] = action
        rewards[i, :, 0] = [0] * batch_size
        terminals[i, :, 0] = [0] * batch_size
        for j in range(9):
            obs, reward, done, info = env.step(action)
            observations[i, :, j + 1, :] = obs
            action = env.offline_action
            actions[i, :, j + 1] = action
            rewards[i, :, j + 1] = env.offline_reward
            terminals[i, :, j + 1] = done

    print('max', np.max(actions))
    print('actions shape', actions.shape)

    # shuffle
    p = np.random.permutation(epoch)
    observations = observations[p]
    actions = actions[p]
    rewards = rewards[p]
    terminals = terminals[p]

    # reshape
    observations.shape = (epoch * batch_size * 10, -1)
    actions.shape = (epoch * batch_size * 10, -1)
    rewards.shape = (epoch * batch_size * 10,)
    terminals.shape = (epoch * batch_size * 10,)

    dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=False)
    print('get_action_size', dataset.episodes[0].get_action_size())

    # # save as HDF5
    dataset.dump(datasetfile)

def data_generate_rl4rs_b(config, datasetfile):
    if config.get('gpu', 0) < 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    batch_size = config["batch_size"]
    sim = SeqSlateRecEnv(config, state_cls=SeqSlateState)
    env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    epoch = 500000 // batch_size
    max_step = config['max_steps']

    observations = np.zeros((epoch, batch_size, max_step + 1, 256 + 9 + 1), 'float32')
    actions = np.zeros((epoch, batch_size, max_step + 1), 'float32')
    rewards = np.zeros((epoch, batch_size, max_step + 1), 'float32')
    terminals = np.zeros((epoch, batch_size, max_step + 1), 'float32')
    for i in range(epoch):
        print('epoch', i)
        obs = env.reset()
        observations[i, :, 0, :] = obs
        action = env.offline_action
        actions[i, :, 0] = action
        rewards[i, :, 0] = [0] * batch_size
        terminals[i, :, 0] = [0] * batch_size
        for j in range(max_step):
            obs, reward, done, info = env.step(action)
            observations[i, :, j + 1, :] = obs
            action = env.offline_action
            actions[i, :, j + 1] = action
            rewards[i, :, j + 1] = env.offline_reward
            terminals[i, :, j + 1] = done

    print('max', np.max(actions))

    # shuffle
    p = np.random.permutation(epoch)
    observations = observations[p]
    actions = actions[p]
    rewards = rewards[p]
    terminals = terminals[p]

    # reshape
    observations.shape = (epoch * batch_size * (max_step + 1), -1)
    actions.shape = (epoch * batch_size * (max_step + 1), 1)
    rewards.shape = (epoch * batch_size * (max_step + 1),)
    terminals.shape = (epoch * batch_size * (max_step + 1),)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # # save as HDF5
    dataset.dump(datasetfile)


def data_generate_rl4rs_b_conti(config, datasetfile):
    config["support_conti_env"] = 1
    if config.get('gpu', 0) < 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    if config.get("support_onehot_action", False):
        config['action_emb_size'] = config["action_size"]

    batch_size = config["batch_size"]
    sim = SeqSlateRecEnv(config, state_cls=SeqSlateState)
    env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    epoch = 500000 // batch_size
    max_step = config['max_steps']

    observations = np.zeros((epoch, batch_size, max_step + 1, 256 + 9 + 1), 'float32')
    actions = np.zeros((epoch, batch_size, max_step + 1, config['action_emb_size']), 'float32')
    rewards = np.zeros((epoch, batch_size, max_step + 1), 'float32')
    terminals = np.zeros((epoch, batch_size, max_step + 1), 'float32')
    for i in range(epoch):
        print('epoch', i)
        obs = env.reset()
        observations[i, :, 0, :] = obs
        action = env.offline_action
        actions[i, :, 0] = action
        rewards[i, :, 0] = [0] * batch_size
        terminals[i, :, 0] = [0] * batch_size
        for j in range(max_step):
            obs, reward, done, info = env.step(action)
            observations[i, :, j + 1, :] = obs
            action = env.offline_action
            actions[i, :, j + 1] = action
            rewards[i, :, j + 1] = env.offline_reward
            terminals[i, :, j + 1] = done

    print('max', np.max(actions))

    # shuffle
    p = np.random.permutation(epoch)
    observations = observations[p]
    actions = actions[p]
    rewards = rewards[p]
    terminals = terminals[p]

    # reshape
    observations.shape = (epoch * batch_size * (max_step + 1), -1)
    actions.shape = (epoch * batch_size * (max_step + 1), config['action_emb_size'])
    rewards.shape = (epoch * batch_size * (max_step + 1),)
    terminals.shape = (epoch * batch_size * (max_step + 1),)

    dataset = MDPDataset(observations, actions, rewards, terminals, discrete_action=False)

    # # save as HDF5
    dataset.dump(datasetfile)


def d3rlpy_eval(eval_episodes, policy: policy_model, soft_opc_score=90):
    if isinstance(policy.policy, d3rlpy.algos.DiscreteBC):
        scorers = {
            'discrete_action_match': d3rlpy_scorer.discrete_action_match_scorer,
        }
        for name, scorer in scorers.items():
            test_score = scorer(policy, eval_episodes)
            print(name, ' ', test_score)
    else:
        scorers = {
            'discrete_action_match': d3rlpy_scorer.discrete_action_match_scorer,
            'soft_opc': d3rlpy_scorer.soft_opc_scorer(soft_opc_score)
        }
        for name, scorer in scorers.items():
            test_score = scorer(policy, eval_episodes)
            print(name, ' ', test_score)


def evaluate(config, policy: policy_model):
    eval_config = config.copy()
    eval_config["is_eval"] = True
    eval_config["batch_size"] = 2048
    max_steps = eval_config["max_steps"]
    if eval_config['env'] == 'SlateRecEnv-v0':
        sim = SlateRecEnv(eval_config, state_cls=SlateState)
        eval_env = gym.make('SlateRecEnv-v0', recsim=sim)
    elif eval_config['env'] == 'SeqSlateRecEnv-v0':
        sim = SeqSlateRecEnv(eval_config, state_cls=SeqSlateState)
        eval_env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    else:
        assert eval_config['env'] in ('SlateRecEnv-v0', 'SeqSlateRecEnv-v0')

    epoch = 4
    episode_rewards, prev_actions = [], []
    for i in range(epoch):
        obs = eval_env.reset()
        episode_reward = []
        print('test batch at ', i)
        for j in range(max_steps):
            action = policy.predict_with_mask(obs)
            obs, reward, done, info = eval_env.step(action)
            episode_reward.append(reward)
            prev_actions.append(action)

        episode_reward = np.sum(np.array(episode_reward), axis=0)
        episode_rewards.append(episode_reward)

    episode_rewards = np.array(episode_rewards)
    print('reward max min', np.max(episode_rewards), np.min(episode_rewards))
    print('avg reward', np.sum(episode_rewards) / eval_config['batch_size'] / epoch)
