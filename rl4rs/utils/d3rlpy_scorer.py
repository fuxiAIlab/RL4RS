import numpy as np
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from d3rlpy.metrics.scorer import AlgoProtocol, _make_batches
from d3rlpy.dataset import Episode
from rl4rs.policy.policy_model import policy_model

WINDOW_SIZE = 1024


# modify from https://github.com/takuseno/d3rlpy/blob/master/d3rlpy/metrics/scorer.py
def soft_opc_scorer(
        return_threshold: float,
) -> Callable[[policy_model, List[Episode]], float]:
    r"""Returns Soft Off-Policy Classification metrics.

    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer funciton is evaluating gaps of action-value
    estimation between the success episodes and the all episodes.
    If the learned Q-function is optimal, action-values in success episodes
    are expected to be higher than the others.
    The success episode is defined as an episode with a return above the given
    threshold.

    .. math::

        \mathbb{E}_{s, a \sim D_{success}} [Q(s, a)]
            - \mathbb{E}_{s, a \sim D} [Q(s, a)]

    .. code-block:: python

        from d3rlpy.datasets import get_cartpole
        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import soft_opc_scorer
        from sklearn.model_selection import train_test_split

        dataset, _ = get_cartpole()
        train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

        scorer = soft_opc_scorer(return_threshold=180)

        dqn = DQN()
        dqn.fit(train_episodes,
                eval_episodes=test_episodes,
                scorers={'soft_opc': scorer})

    References:
        * `Irpan et al., Off-Policy Evaluation via Off-Policy Classification.
          <https://arxiv.org/abs/1906.01624>`_

    Args:
        return_threshold: threshold of success episodes.

    Returns:
        scorer function.

    """

    def scorer(algo: policy_model, episodes: List[Episode]) -> float:
        success_values = []
        all_values = []
        for episode in episodes:
            is_success = episode.compute_return() >= return_threshold
            for batch in _make_batches(episode, WINDOW_SIZE, algo.policy.n_frames):
                values = algo.predict_q(batch.observations, batch.actions)
                values = cast(np.ndarray, values)
                all_values += values.reshape(-1).tolist()
                if is_success:
                    success_values += values.reshape(-1).tolist()
        return float(np.mean(success_values) - np.mean(all_values))

    return scorer


def dynamics_reward_prediction_mean_error_scorer(
        dynamics: policy_model, episodes: List[Episode]
) -> float:
    r"""Returns MSE of reward prediction (in negative scale).

    This metrics suggests how dynamics model is generalized to test sets.
    If the MSE is large, the dynamics model are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1} \sim D} [(r_{t+1} - r')]

    where :math:`r' \sim T(s_t, a_t)`.

    Args:
        dynamics: dynamics model.
        episodes: list of episodes.

    Returns:
        negative mean squared error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.policy.n_frames):
            pred = dynamics.predict_q(batch.observations, batch.actions)
            rewards = batch.next_rewards
            errors = (rewards - pred[1]).reshape(-1)
            total_errors += errors.tolist()
    # smaller is better
    return float(np.mean(total_errors))


def dynamics_reward_prediction_abs_mean_error_scorer(
        dynamics: policy_model, episodes: List[Episode]
) -> float:
    r"""Returns MSE of reward prediction (in negative scale).

    This metrics suggests how dynamics model is generalized to test sets.
    If the MSE is large, the dynamics model are overfitting.

    .. math::

        \mathbb{E}_{s_t, a_t, r_{t+1} \sim D} [abs(r_{t+1} - r')]

    where :math:`r' \sim T(s_t, a_t)`.

    Args:
        dynamics: dynamics model.
        episodes: list of episodes.

    Returns:
        negative mean squared error.

    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, dynamics.policy.n_frames):
            pred = dynamics.predict_q(batch.observations, batch.actions)
            rewards = batch.next_rewards
            errors = np.abs(rewards - pred[1]).reshape(-1)
            total_errors += errors.tolist()
    # smaller is better
    return float(np.mean(total_errors))

def discrete_action_match_scorer(
        algo: policy_model, episodes: List[Episode]
) -> float:
    r"""Returns percentage of identical actions between algorithm and dataset.

    This metrics suggests how different the greedy-policy is from the given
    episodes in discrete action-space.
    If the given episdoes are near-optimal, the large percentage would be
    better.

    .. math::

        \frac{1}{N} \sum^N \parallel
            \{a_t = \text{argmax}_a Q_\theta (s_t, a)\}

    Args:
        algo: algorithm.
        episodes: list of episodes.

    Returns:
        percentage of identical actions.

    """
    total_matches = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.policy.n_frames):
            actions = algo.predict_with_mask(batch.observations)
            match = (batch.actions.reshape(-1) == actions).tolist()
            total_matches += match
    return float(np.mean(total_matches))