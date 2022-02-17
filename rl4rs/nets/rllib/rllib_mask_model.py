import tensorflow as tf
from rl4rs.nets.rllib.rllib_rawstate_model import TFModelWithRawState
from ray.rllib.examples.models.parametric_actions_model import \
    ParametricActionsModel


def getMaskActionsModel(true_obs_shape, action_size):
    class MyMaskActionsModel(ParametricActionsModel):
        """Parametric action model that handles the dot product and masking.

        This assumes the outputs are logits for a single Categorical action dist.
        Getting this to work with a more complex output (e.g., if the action space
        is a tuple of several distributions) is also possible but left as an
        exercise to the reader.
        """

        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     **kw):
            config = {
                # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
                # These are used if no custom model is specified and the input space is 1D.
                # Number of hidden layers to be used.
                "fcnet_hiddens": [64],
                # Activation function descriptor.
                # Supported values are: "tanh", "relu", "swish" (or "silu"),
                # "linear" (or None).
                # "fcnet_activation": "linear",
                # "no_final_linear": True,
                "vf_share_layers": True,
            }
            model_config = dict(model_config, **config)
            super(MyMaskActionsModel, self).__init__(
                obs_space, action_space, num_outputs, model_config, name, true_obs_shape, action_embed_size=action_size, **kw)
            print('MyMaskActionsModel', self.action_embed_model.model_config)

        def forward(self, input_dict, state, seq_lens):
            # Extract the available actions tensor from the observation.
            # avail_actions = input_dict["obs"]["avail_actions"]
            action_mask = input_dict["obs"]["action_mask"]

            # Compute the predicted action embedding
            action_embed, _ = self.action_embed_model({
                "obs": input_dict["obs"]["obs"]
            })
            # action_values = self.action_embed_model.value_function()
            # print(tf.shape(action_embed), action_embed)

            # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
            # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
            # intent_vector = tf.expand_dims(action_embed, 1)

            # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
            # action_prob = tf.nn.softmax(action_embed)

            # Mask out invalid actions (use tf.float32.min for stability)
            inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
            return action_embed + inf_mask, state

    return MyMaskActionsModel


def getMaskActionsModelWithRawState(config, action_size):
    config = config

    class MyMaskActionsModelWithRawState(ParametricActionsModel):
        """Parametric action model that handles the dot product and masking.

        This assumes the outputs are logits for a single Categorical action dist.
        Getting this to work with a more complex output (e.g., if the action space
        is a tuple of several distributions) is also possible but left as an
        exercise to the reader.
        """

        def __init__(self,
                     obs_space,
                     action_space,
                     num_outputs,
                     model_config,
                     name,
                     **kw):
            # model_config = dict(model_config, **config)
            super(MyMaskActionsModelWithRawState, self).__init__(
                obs_space, action_space, num_outputs, model_config, name, action_embed_size=action_size, **kw)
            print('MyMaskActionsModelWithRawStateModel', self.action_embed_model.model_config)
            self.action_embed_model = TFModelWithRawState(
                obs_space, action_space, action_size,
                model_config, name + "_action_embed", config = config)

        def forward(self, input_dict, state, seq_lens):
            # Extract the available actions tensor from the observation.
            # avail_actions = input_dict["obs"]["avail_actions"]
            action_mask = input_dict["obs"]["action_mask"]

            # Compute the predicted action embedding
            action_embed, _ = self.action_embed_model(input_dict)
            # action_values = self.action_embed_model.value_function()
            # print(tf.shape(action_embed), action_embed)

            # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
            # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
            # intent_vector = tf.expand_dims(action_embed, 1)

            # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
            # action_prob = tf.nn.softmax(action_embed)

            # Mask out invalid actions (use tf.float32.min for stability)
            inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
            return action_embed + inf_mask, state

    return MyMaskActionsModelWithRawState
