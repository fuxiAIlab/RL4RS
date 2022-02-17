import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from rl4rs.nets import utils


def custom_loss(external_loss):
    def loss(y_true, y_pred):
        return 0.1 * tf.keras.losses.binary_crossentropy(y_true, y_pred) + external_loss

    return loss


def my_loss_fn(y_true, y_pred):
    item_scores_exp = tf.exp(y_pred)
    item_scores_click = tf.einsum('ij,ij->ij', y_pred, tf.cast(y_true, tf.float32))
    return -tf.log(tf.reduce_sum(tf.exp(item_scores_click), axis=1) + 1) \
           + tf.log(tf.reduce_sum(item_scores_exp, axis=1) + 1)


def my_metrics(y_true, y_pred):
    score = tf.einsum('ij,ij->ij', y_pred, 1 - tf.cast(y_true, tf.float32))
    return tf.reduce_sum(score, 1)


def my_mean_metrics(y_true, y_pred):
    return tf.reduce_mean(y_pred, 1)


def my_max_metrics(y_true, y_pred):
    return tf.reduce_max(y_pred, 1)


def my_min_metrics(y_true, y_pred):
    return tf.reduce_min(y_pred, 1)


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

    feature_omit = layers.Lambda(lambda x: x[:, :-1])
    category_feature_input_slate = feature_omit(category_feature_input)
    config['category_feature_num'] = config['category_feature_num'] - 1

    category_feature = utils.id_input_processing(category_feature_input_slate, config)
    dense_feature = utils.dense_input_processing(dense_feature_input, config)
    sequence_feature = utils.sequence_input_concat(sequence_feature_input, config)

    all_feature = layers.Concatenate(axis=-1)(
        [sequence_feature, dense_feature, category_feature]
    )
    item_scores = layers.Dense(9, activation='sigmoid')(all_feature)
    item_scores_norm = layers.Softmax()(item_scores)
    item_scores_no_click = tf.einsum('ij,ij->ij',
                                     item_scores_norm,
                                     1 - tf.cast(slate_label_input, tf.float32))
    loss3 = tf.reduce_sum(item_scores_no_click, axis=1)

    model = Model(inputs=[sequence_feature_input,
                          dense_feature_input,
                          category_feature_input,
                          slate_label_input],
                  outputs=[item_scores])
    model.compile(optimizer='adam',
                  loss=custom_loss(loss3),
                  metrics=[
                      tf.keras.metrics.AUC(),
                      tf.keras.metrics.Precision(),
                      tf.keras.metrics.Recall()])
    return model
