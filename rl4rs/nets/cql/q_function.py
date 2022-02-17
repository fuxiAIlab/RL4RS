from typing import Optional, cast

import torch
import torch.nn.functional as F
from torch import nn
from typing import Any, ClassVar, Dict, Type
from d3rlpy.models.torch.encoders import Encoder
from d3rlpy.models.torch.q_functions.base import DiscreteQFunction
from d3rlpy.models.torch.q_functions.utility import compute_huber_loss, compute_reduce, pick_value_by_action
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import EncoderWithAction, ContinuousMeanQFunction


class CustomDiscreteMeanQFunction(DiscreteQFunction, nn.Module):  # type: ignore
    _action_size: int
    _encoder: Encoder
    _fc: nn.Linear

    def __init__(self, encoder: Encoder, action_size: int):
        super().__init__()
        self._action_size = action_size
        self._encoder = encoder
        # self._fc = nn.Linear(encoder.get_feature_size(), action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._encoder(x))

    def compute_error(
            self,
            obs_t: torch.Tensor,
            act_t: torch.Tensor,
            rew_tp1: torch.Tensor,
            q_tp1: torch.Tensor,
            ter_tp1: torch.Tensor,
            gamma: float = 0.99,
            reduction: str = "mean",
    ) -> torch.Tensor:
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        q_t = (self.forward(obs_t) * one_hot.float()).sum(dim=1, keepdim=True)
        y = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)
        loss = compute_huber_loss(q_t, y)
        return compute_reduce(loss, reduction)

    def compute_target(
            self, x: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if action is None:
            return self.forward(x)
        # q=pick_value_by_action(self.forward(x), action, keepdim=True)
        values = self.forward(x)
        action_size = values.shape[1]
        one_hot = F.one_hot(action.view(-1), num_classes=action_size)
        masked_values = values * cast(torch.Tensor, one_hot.float())
        q = masked_values.sum(dim=1, keepdim=True)
        # assert torch.min(q)>-100
        return q

    @property
    def action_size(self) -> int:
        return self._action_size

    @property
    def encoder(self) -> Encoder:
        return self._encoder


class CustomMeanQFunctionFactory(QFunctionFactory):
    TYPE: ClassVar[str] = "mean"

    def __init__(self, bootstrap: bool = False, share_encoder: bool = False):
        super().__init__(bootstrap, share_encoder)

    def create_discrete(
            self,
            encoder: Encoder,
            action_size: int,
    ) -> CustomDiscreteMeanQFunction:
        return CustomDiscreteMeanQFunction(encoder, action_size)

    def create_continuous(
            self,
            encoder: EncoderWithAction,
    ) -> ContinuousMeanQFunction:
        return ContinuousMeanQFunction(encoder)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "bootstrap": self._bootstrap,
            "share_encoder": self._share_encoder,
        }
