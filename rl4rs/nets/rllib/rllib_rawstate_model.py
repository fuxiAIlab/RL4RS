import gym
from gym.spaces import Dict
from rl4rs.nets import utils
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.framework import try_import_tf, try_import_torch

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


def getTFModelWithRawState(config):
    config = config

    class MyTFModelWithRawState(TFModelWithRawState):
        def __init__(self, obs_space, action_space, num_outputs, model_config,
                     name):
            super(MyTFModelWithRawState, self).__init__(
                obs_space, action_space, num_outputs, model_config, name, config=config)

    return MyTFModelWithRawState


class TFModelWithRawState(TFModelV2):
    """Implements the `.action_model` branch required above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, config):
        obs_space = obs_space.original_space
        super(TFModelWithRawState, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)
        if not (isinstance(obs_space, Dict) and obs_space['category_feature'] \
                and obs_space['dense_feature'] and obs_space['sequence_feature']):
            raise ValueError("""This model only supports the Dict{'category_feature':[], 
                'dense_feature':[], 'sequence_feature':[]} obs space""")
        activation = model_config.get("fcnet_activation", "linear")
        activation = get_activation_fn(activation)
        no_final_linear = model_config.get("no_final_linear", False)
        # Inputs
        category_feature_input = tf.keras.layers.Input(
            shape=obs_space['category_feature'].shape, name="obs_category_input")
        dense_feature_input = tf.keras.layers.Input(
            shape=obs_space['dense_feature'].shape, name="obs_dense_input")
        sequence_feature_input = tf.keras.layers.Input(
            shape=obs_space['sequence_feature'].shape, name="obs_sequence_input")

        slice_layer = tf.keras.layers.Lambda(lambda x: x[0][:, x[1]:])
        category_feature = utils.id_input_processing(category_feature_input, config)
        dense_feature = utils.dense_input_processing(dense_feature_input, config)
        sequence_feature = utils.sequence_input_concat(sequence_feature_input, config)
        all_feature = tf.keras.layers.Concatenate(axis=-1)([sequence_feature, dense_feature, category_feature])
        context = tf.keras.layers.Dense(256, activation=tf.keras.layers.ELU())(all_feature)
        model_out = None
        if no_final_linear and num_outputs:
            model_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(context)
        else:
            model_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=None,
                kernel_initializer=normc_initializer(0.01))(context)

        # V(s)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(context)

        # Base layers
        self.base_model = tf.keras.Model([category_feature_input, dense_feature_input, sequence_feature_input], [model_out, value_out])
        self.base_model.summary()

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model([input_dict["obs"]["category_feature"],
                                                      input_dict["obs"]["dense_feature"],
                                                      input_dict["obs"]["sequence_feature"]])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])
