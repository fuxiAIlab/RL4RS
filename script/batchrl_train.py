import os
import gym
import random
import d3rlpy
import sys
import torch
from rl4rs.env.slate import SlateRecEnv, SlateState
from rl4rs.env.seqslate import SeqSlateRecEnv, SeqSlateState
from script import batchrl_trainer
from d3rlpy.dataset import MDPDataset
from rl4rs.nets.cql.encoder import CustomVectorEncoderFactory
from rl4rs.nets.cql.q_function import CustomMeanQFunctionFactory
from script.offline_evaluation import ope_eval
from rl4rs.policy.behavior_model import behavior_model

algo = sys.argv[1]
stage = sys.argv[2]
extra_config = eval(sys.argv[3]) if len(sys.argv) >= 4 else {}

config = {"epoch": 4, "maxlen": 64, "batch_size": 2048, "action_size": 284, "class_num": 2, "dense_feature_num": 432,
          "category_feature_num": 21, "category_hash_size": 100000, "seq_num": 2, "emb_size": 128,
          "hidden_units": 128, "max_steps": 9, "sample_file": '../dataset/rl4rs_dataset_a_shuf.csv',
          "model_file": "../output/rl4rs_dataset_a_dnn/model", 'gpu': True, "page_items": 9,
          "iteminfo_file": '../dataset/item_info.csv', "support_d3rl_mask": True, "is_eval": True,
          "CQL_alpha": 1, 'env': 'SlateRecEnv-v0', 'trial_name': 'a_all'}

config = dict(config, **extra_config)

if config['env'] == 'SeqSlateRecEnv-v0':
    config['max_steps'] = 36
    location_mask, special_items = SeqSlateState.get_mask_from_file(config['iteminfo_file'], config['action_size'])
    config['location_mask'] = location_mask
    config['special_items'] = special_items
elif config['env'] == 'SlateRecEnv-v0':
    location_mask, special_items = SlateState.get_mask_from_file(config['iteminfo_file'], config['action_size'])
    config['location_mask'] = location_mask
    config['special_items'] = special_items
else:
    assert config['env'] in ('SlateRecEnv-v0', 'SeqSlateRecEnv-v0')

if not config.get('gpu', True):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    torch.cuda.is_available = lambda: False
    print('CUDA_VISIBLE_DEVICES', torch.cuda.is_available())

trail_name = config['env'] + '_' + config['trial_name'] + '.h5'
dataset_dir = os.environ['rl4rs_dataset_dir']
output_dir = os.environ['rl4rs_output_dir']
dataset_save_path = dataset_dir + '/' + trail_name
model_save_path = output_dir + '/' + algo + '_' + trail_name
scaler = None

print(trail_name, config)

if algo == 'BC':
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
        q_func_factory=CustomMeanQFunctionFactory(share_encoder=True),
        encoder_factory=encoder_factory,
        gamma=1.0,
        alpha=config["CQL_alpha"],
        reward_scaler='standard',
        use_gpu=config['gpu']
    )
else:
    assert algo in ('BC', 'BCQ', 'CQL')

if stage == 'dataset_generate':
    if config['env'] == 'SlateRecEnv-v0':
        batchrl_trainer.data_generate_rl4rs_a(config, dataset_save_path)
    elif config['env'] == 'SeqSlateRecEnv-v0':
        batchrl_trainer.data_generate_rl4rs_b(config, dataset_save_path)
    else:
        batchrl_trainer.data_generate_rl4rs_a(config, dataset_save_path)
        assert config['env'] in ('SlateRecEnv-v0', 'SeqSlateRecEnv-v0')

if stage == 'train':
    dataset = MDPDataset.load(dataset_save_path)
    model.fit(dataset,
              eval_episodes=dataset.episodes[-3000:],
              n_epochs=config['epoch'],
              show_progress=False)
    model.save_model(model_save_path)

if stage == 'eval':
    default_soft_opc_score = 90 \
        if config['env'] == 'SlateRecEnv-v0' \
        else 90 * 2
    soft_opc_score = config.get('soft_opc_score', default_soft_opc_score)
    dataset = MDPDataset.load(dataset_save_path)
    model.build_with_dataset(dataset)
    model.load_model(model_save_path)
    eval_episodes = random.sample(dataset.episodes, 2048 * 4)
    policy = policy_model(model, config=config)
    batchrl_trainer.d3rlpy_eval(eval_episodes, policy, soft_opc_score)
    batchrl_trainer.evaluate(config, policy)

if stage == 'ope':
    dataset_dir = os.environ['rl4rs_dataset_dir']
    sample_model = behavior_model(config, modelfile=dataset_dir + '/logged_policy.h5')
    dataset = MDPDataset.load(dataset_save_path)
    model.build_with_dataset(dataset)
    model.load_model(model_save_path)
    if model.reward_scaler is not None:
        model.reward_scaler.fit(dataset)
        print('reward_scaler_mean', model.reward_scaler._mean)
        print('reward_scaler_std', model.reward_scaler._std)
    eval_config = config.copy()
    eval_config["is_eval"] = True
    eval_config["batch_size"] = 2048
    eval_config["epoch"] = 1
    if config['env'] == 'SeqSlateRecEnv-v0':
        config['max_steps'] = 36
        sim = SeqSlateRecEnv(eval_config, state_cls=SeqSlateState)
        eval_env = gym.make('SeqSlateRecEnv-v0', recsim=sim)
    else:
        sim = SlateRecEnv(eval_config, state_cls=SlateState)
        eval_env = gym.make('SlateRecEnv-v0', recsim=sim)
    ope_eval(eval_config, eval_env, model, sample_model=sample_model)
