import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.pg as pg
import ray.rllib.agents.impala as impala
import ray.rllib.agents.ddpg as ddpg


def get_rl_model(algo, rllib_config):
    trainer = None
    if algo == "PPO":
        trainer = ppo.PPOTrainer(config=rllib_config, env="rllibEnv-v0")
        print('trainer_default_config', trainer._default_config)
    elif algo == "DQN":
        trainer = dqn.DQNTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "A2C":
        trainer = a3c.A2CTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "A3C":
        trainer = a3c.A3CTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "PG":
        trainer = pg.PGTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "DDPG":
        trainer = ddpg.DDPGTrainer(config=rllib_config, env="rllibEnv-v0")
    elif algo == "IMPALA":
        trainer = impala.ImpalaTrainer(config=rllib_config, env="rllibEnv-v0")
        print('trainer_default_config', trainer._default_config)
    else:
        assert algo in ("PPO", "DQN", "A2C", "A3C", "PG", "IMPALA")
    return trainer
