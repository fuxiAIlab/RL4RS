import torch
import torch.nn as nn
import numpy as np
import copy
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Type, Union
from d3rlpy.models.encoders import EncoderFactory, Encoder, VectorEncoderWithAction, _create_activation, VectorEncoder


class CustomVectorEncoder(VectorEncoder):

    def __init__(
            self,
            config,
            action_size,
            mask_size,
            with_q,
            observation_shape: Sequence[int],
            hidden_units: Optional[Sequence[int]] = None,
            use_batch_norm: bool = False,
            dropout_rate: Optional[float] = None,
            use_dense: bool = False,
            activation: nn.Module = nn.ReLU(),
    ):
        super().__init__(observation_shape, hidden_units, use_batch_norm, dropout_rate, use_dense, activation)
        self.action_size = action_size
        self.mask_size = mask_size
        self.with_q = with_q
        self.emb_size = 32
        self.emb_layer = nn.Embedding(action_size, self.emb_size)
        self.fc2 = nn.Linear(self._feature_size + self.emb_size * mask_size, action_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        location_mask = config['location_mask']
        self.special_items = config['special_items']
        self.location_mask = torch.tensor(location_mask, device=self.device)

    def get_feature_size(self) -> int:
        if not self.with_q:
            return self._feature_size + self.emb_size * self.mask_size
        else:
            return self.action_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # mask
        prev_actions = x[:, -self.mask_size:-1].to(torch.long)
        cur_step = x[:, -1].to(torch.long)
        x_mask_layer = cur_step % 9 // 3
        mask = self.location_mask[x_mask_layer]
        for i in range(self.mask_size-1):
            mask[range(batch_size), prev_actions[:, i]] = 0
        h = self._fc_encode(x)
        if self._use_batch_norm:
            h = self._bns[-1](h)
        if self._dropout_rate is not None:
            h = self._dropouts[-1](h)
        prev_action_emb = nn.Flatten()(self.emb_layer(x[:, -self.mask_size:].to(torch.long)))
        h = torch.cat([h, prev_action_emb], dim=-1)
        if self.with_q:
            h = self.fc2(h)
            action_mask = mask < 0.01
            # h[action_mask] = -2 ** 15
            h[action_mask] = 0
            for i in range(batch_size):
                if len(np.intersect1d(prev_actions[i].cpu().numpy(), self.special_items)) > 0:
                    h[i][self.special_items] = 0
                    # h[i][self.special_items] = -2 ** 15
        return h


class CustomVectorEncoderFactory(EncoderFactory):
    TYPE: ClassVar[str] = "vector"
    _hidden_units: Sequence[int]
    _activation: str
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool

    def __init__(
            self,
            config,
            action_size,
            mask_size,
            with_q=False,
            hidden_units: Optional[Sequence[int]] = None,
            activation: str = "relu",
            use_batch_norm: bool = False,
            dropout_rate: Optional[float] = None,
            use_dense: bool = False,
    ):
        self.config = config
        self.action_size = action_size
        self.mask_size = mask_size
        self.with_q = with_q
        if hidden_units is None:
            self._hidden_units = [256]
        else:
            self._hidden_units = hidden_units
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._use_dense = use_dense

    def create(self, observation_shape: Sequence[int]) -> CustomVectorEncoder:
        assert len(observation_shape) == 1
        return CustomVectorEncoder(
            config=self.config,
            action_size=self.action_size,
            mask_size=self.mask_size,
            with_q=self.with_q,
            observation_shape=observation_shape,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            activation=_create_activation(self._activation),
        )

    def create_with_action(
            self,
            observation_shape: Sequence[int],
            action_size: int,
            discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        assert len(observation_shape) == 1
        return VectorEncoderWithAction(
            observation_shape=observation_shape,
            action_size=action_size,
            hidden_units=self._hidden_units,
            use_batch_norm=self._use_batch_norm,
            dropout_rate=self._dropout_rate,
            use_dense=self._use_dense,
            discrete_action=discrete_action,
            activation=_create_activation(self._activation),
        )

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            hidden_units = copy.deepcopy(self._hidden_units)
        else:
            hidden_units = self._hidden_units
        params = {
            "hidden_units": hidden_units,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "dropout_rate": self._dropout_rate,
            "use_dense": self._use_dense,
        }
        return params
