import numpy as np
# import scipy
import scipy.stats


# modify from https://mars-gym.readthedocs.io/en/latest/quick_start.html#off-policy-metrics

def _calc_sequential_weigths(policy_prob, behavior_prob, weighted=False, a_min=None, a_max=None):
    # behavior_prob: Coleta
    # policy_prob: Avaliação
    #
    # Compute the sample weights - propensity ratios
    probs = np.array(policy_prob) / np.array(behavior_prob)
    rho = np.clip(probs, a_min=a_min, a_max=a_max).cumprod(1)
    if weighted:
        weight = np.sum(rho, axis=0)
    else:
        weight = len(policy_prob)
    ws = rho / weight
    return np.clip(ws, a_min=a_min, a_max=a_max)


def _calc_sample_weigths(policy_prob, behavior_prob, a_min=None, a_max=None):
    # behavior_prob: Coleta
    # policy_prob: Avaliação
    #
    # Compute the sample weights - propensity ratios
    p_ratio = np.array(policy_prob) / np.array(behavior_prob)

    if a_min is not None:
        p_ratio = np.clip(p_ratio, a_min=a_min, a_max=a_max)

    # Effective sample size for E_t estimate (from A. Owen)
    n_e = len(policy_prob) * (np.mean(p_ratio) ** 2) / (p_ratio ** 2).mean()

    # Critical value from t-distribution as we have unknown variance
    alpha = 0.00125
    cv = scipy.stats.t.ppf(1 - alpha, df=int(n_e) - 1)

    return p_ratio, n_e, cv


def eval_DM(policy, obs):
    return policy(obs)


def eval_IPS(rewards, policy_prob, behavior_prob):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(policy_prob, behavior_prob)
    ###############
    # VANILLA IPS #
    ###############
    # Expected reward for pi_t
    E_t = np.mean(rewards * p_ratio)

    # Variance of the estimate
    var = ((rewards * p_ratio - E_t) ** 2).mean()
    stddev = np.sqrt(var)

    # C.I. assuming unknown variance - use t-distribution and effective sample size
    c = cv * stddev / np.sqrt(int(n_e))
    min_bound = E_t - c
    max_bound = E_t + c

    result = (E_t, c)  # 0.025, 0.500, 0.975
    return result


def eval_CIPS(rewards, policy_prob, behavior_prob):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(policy_prob, behavior_prob, a_min=0.1, a_max=10)

    ##############
    # CAPPED IPS #
    ##############
    # Cap ratios
    p_ratio_capped = np.clip(p_ratio, a_min=0.1, a_max=10)

    # Expected reward for pi_t
    E_t_capped = np.mean(rewards * p_ratio_capped)

    # Variance of the estimate
    var_capped = ((rewards * p_ratio_capped - E_t_capped) ** 2).mean()
    stddev_capped = np.sqrt(var_capped)

    # C.I. assuming unknown variance - use t-distribution and effective sample size
    c = cv * stddev_capped / np.sqrt(int(n_e))

    min_bound_capped = E_t_capped - c
    max_bound_capped = E_t_capped + c

    result = (E_t_capped, c)  # 0.025, 0.500, 0.975

    return result


def eval_SNIPS(rewards, policy_prob, behavior_prob):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(policy_prob, behavior_prob, a_min=0.1, a_max=10)

    ##############
    # NORMED IPS #
    ##############
    # Expected reward for pi_t
    E_t_normed = np.sum(rewards * p_ratio) / np.sum(p_ratio)

    # Variance of the estimate
    var_normed = np.sum(((rewards - E_t_normed) ** 2) * (p_ratio ** 2)) / (
            p_ratio.sum() ** 2
    )
    stddev_normed = np.sqrt(var_normed)

    # C.I. assuming unknown variance - use t-distribution and effective sample size
    c = cv * stddev_normed / np.sqrt(int(n_e))

    min_bound_normed = E_t_normed - c
    max_bound_normed = E_t_normed + c

    # Store result
    result = (E_t_normed, c)  # 0.025, 0.500, 0.975

    return result


def eval_WIPS(step_rewards, policy_prob, behavior_prob, gamma=1.0):
    batch_size = len(step_rewards)
    steps = len(step_rewards[0])
    w_t = []

    # calculate importance ratios
    p = _calc_sequential_weigths(policy_prob, behavior_prob, a_min=0.1, a_max=10)

    for i in range(steps):
        w_t.append(np.average(p[:, :i + 1], axis=1))
    w_t = np.array(w_t).swapaxes(0, 1)
    # calculate stepwise weighted IS estimate
    V_prev, V_step_WIS = 0.0, 0.0
    for t in range(steps):
        V_prev += np.sum(step_rewards[:, t] * gamma ** t)
        V_step_WIS += np.sum(p[:, t] / w_t[:, t] * step_rewards[:, t] * gamma ** t)
    # print('WIPS', p[:, -1], w_t[:, -1], np.max(p[:, -1] / w_t[:, -1]), step_rewards[:, -1])
    return V_step_WIS / np.clip(V_prev, a_min=1e-8, a_max=None), 0


def eval_doubly_robust(
        action_rhat_rewards, state_rewards, rewards, policy_prob, behavior_prob
):
    # Calculate Sample Weigths
    p_ratio, n_e, cv = _calc_sample_weigths(policy_prob, behavior_prob, a_min=0.1, a_max=10)

    #################
    # Roubly Robust #
    #################

    dr = state_rewards + (p_ratio * (rewards - action_rhat_rewards))

    confidence = 0.95
    n = len(dr)
    m, se = np.mean(dr), scipy.stats.sem(dr)
    # h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    # print('dr', action_rhat_rewards[:2], p_ratio[:2], rewards[:2], m)
    return m / np.average(rewards), se


def eval_seq_doubly_robust(
        action_rhat_rewards, state_rewards, rewards, policy_prob, behavior_prob
):
    # Calculate Sample Weigths
    ws = _calc_sequential_weigths(policy_prob, behavior_prob, a_min=0.1, a_max=10)

    dr = np.zeros((len(action_rhat_rewards)))
    steps = len(action_rhat_rewards[0])
    for i in range(steps):
        t = steps - i - 1
        dr = state_rewards[:, t] + ws[:, t] * (rewards[:, t] + dr - action_rhat_rewards[:, t])

    #################
    # Roubly Robust #
    #################
    # dr = action_rhat_rewards + (p_ratio * (rewards - action_rhat_rewards))
    # estimate = ws * (rewards - action_rhat_rewards) +  state_rewards
    # print('sdr', dr, np.average(dr), np.average(rewards))

    return np.average(dr) / np.mean(np.sum(rewards, axis=1)), 0


if __name__ == '__main__':
    batch_size = 10
    max_steps = 9
    off_rewards_sum = np.ones(batch_size, )
    action_probs_mul = np.random.random((batch_size,))
    behavior_probs_mul = np.random.random((batch_size,))
    episode_reward = np.random.random((batch_size,)) * 2
    off_rewards = np.ones((batch_size, max_steps))
    action_probs = np.random.random((batch_size, max_steps))
    behavior_probs = np.random.random((batch_size, max_steps))
    rewards_hat = np.random.random((batch_size, max_steps))
    state_reward = np.ones((batch_size, max_steps))

    ips = eval_IPS(off_rewards_sum, action_probs_mul, behavior_probs_mul)
    cips = eval_CIPS(off_rewards_sum, action_probs_mul, behavior_probs_mul)
    snips = eval_SNIPS(off_rewards_sum, action_probs_mul, behavior_probs_mul)
    dr = eval_doubly_robust(episode_reward, off_rewards_sum, action_probs_mul, behavior_probs_mul)
    # step-wise
    sips = eval_WIPS(off_rewards, action_probs, behavior_probs)
    sdr = eval_seq_doubly_robust(rewards_hat, state_reward, off_rewards, action_probs, behavior_probs)
    print(ips, cips, snips, dr, sips, sdr, sep='\n')
