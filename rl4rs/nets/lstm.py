import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from rl4rs.nets import utils


def get_model(config):
    maxlen = config['maxlen']
    dense_feature_num = config['dense_feature_num']
    category_feature_num = config['category_feature_num']
    class_num = config['class_num']
    seq_num = config['seq_num']

    sequence_feature_input = layers.Input(
        shape=(seq_num, maxlen,), dtype='float32', name='sequence_feature_input'
    )

    dense_feature_input = layers.Input(
        shape=(dense_feature_num,), dtype='float32', name='dense_feature_input'
    )

    category_feature_input = layers.Input(
        shape=(category_feature_num,), dtype='int64', name='category_feature_input'
    )

    slate_label_input = layers.Input(
        shape=(9,), dtype='int64', name='slate_label'
    )

    category_feature = utils.id_input_processing_lstm(category_feature_input, config)
    # category_feature_concat = utils.id_input_processing_concat(category_feature_input, config)
    dense_feature = utils.dense_input_processing(dense_feature_input, config)
    sequence_feature = utils.sequence_input_LSTM(sequence_feature_input, config)
    all_feature = layers.Concatenate(axis=-1)([sequence_feature, dense_feature, category_feature])
    obs = layers.Dense(256, activation=layers.ELU(), name='simulator_obs')(all_feature)
    output = layers.Dense(class_num, activation='softmax', name='simulator_reward')(obs)

    model = Model(inputs=[sequence_feature_input,
                          dense_feature_input,
                          category_feature_input,
                          slate_label_input],
                  outputs=[output])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC', 'acc'])
    return model
