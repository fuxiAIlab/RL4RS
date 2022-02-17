from gym.envs.registration import register

register(
    id='HttpEnv-v0',
    entry_point='rl4rs.server.httpEnv:HttpEnv',
)

register(
    id='SlateRecEnv-v0',
    entry_point='rl4rs.env:RecEnvBase',
)

register(
    id='SeqSlateRecEnv-v0',
    entry_point='rl4rs.env:RecEnvBase',
)
