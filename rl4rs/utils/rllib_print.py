import json
import yaml
from ray.tune.utils.util import SafeFallbackEncoder


def pretty_print(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    result.update(hist_stats=None)  # drop hist_stats from pretty print
    out = {}
    print_keys = ('episode_reward_mean',
                  'episode_reward_min',
                  'timesteps_total',
                  'training_iteration')
    for k, v in result.items():
        if v is not None:
            if k in print_keys:
                out[k] = v
            elif k == 'evaluation':
                out[k] = {
                    'episode_reward_mean': v['episode_reward_mean'],
                    'episode_reward_min': v['episode_reward_min'],
                }
    cleaned = json.dumps(out, cls=SafeFallbackEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)
