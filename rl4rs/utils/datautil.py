from tensorflow.python.data.ops import dataset_ops
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import random


class FeatureUtil(object):

    def __init__(self, config):
        self.config = config
        self.maxlen = config['maxlen']
        self.batch_size = config['batch_size']
        self.class_num = config['class_num']
        self.dense_feature_num = config['dense_feature_num']
        self.category_feature_num = config['category_feature_num']
        self.category_hash_size = config['category_hash_size']
        self.seq_num = self.config['seq_num']

    @classmethod
    def record_split(cls, record):
        timestamp, sess_id, sequence_id, exposed_items, user_feedback, \
            user_seqfeature, user_protrait, item_feature, behavior_id = record.split('@')
        return int(timestamp), \
               int(sess_id), \
               int(sequence_id), \
               list(map(int, exposed_items.split(','))), \
               list(map(int, user_feedback.split(','))), \
               list(map(int, user_seqfeature.split(','))), \
               list(map(float, user_protrait.split(','))), \
               list(map(float, item_feature.replace(';', ',').split(','))), \
               int(behavior_id)

    def feature_extraction(self, data):
        sequence_ids = []
        dense_features = []
        category_features_ids = []
        slate_labels = []
        labels = []
        for record in data:
            role_id, sequence_feature, dense_feature, \
                category_feature, slate_label, label = record
            sequence_id = [
                pad_sequences([xx], maxlen=self.maxlen)[0]
                for xx in sequence_feature[:self.seq_num]
            ]
            sequence_ids.append(sequence_id)
            dense_features.append(dense_feature)
            category_features_ids.append(list(map(int, category_feature)))
            slate_labels.append(slate_label)
            labels.append(label)
        dense_features = pad_sequences(
            dense_features,
            maxlen=self.dense_feature_num,
            dtype='float32',
            padding='post',
            truncating='post'
        )
        category_features_ids = pad_sequences(
            category_features_ids,
            maxlen=self.category_feature_num,
            dtype='int32',
            padding='post',
            truncating='post'
        )
        return (np.array(sequence_ids),
                dense_features,
                category_features_ids,
                np.array(slate_labels)), labels

    def read_tfrecord(self, filename, is_pred=False, is_slate_label=False):
        config = self.config

        def _parse_exmp(serial_exmp):
            context_features = {
                "dense_feature": tf.io.VarLenFeature(dtype=tf.float32),
                "category_feature": tf.io.VarLenFeature(dtype=tf.int64),
                "slate_label": tf.io.VarLenFeature(dtype=tf.int64),
                "label": tf.io.FixedLenFeature([], dtype=tf.int64),
            }

            for seq_num_i in range(config['seq_num']):
                context_features['sequence_id_' + str(seq_num_i)] \
                    = tf.io.VarLenFeature(dtype=tf.int64)

            context_parsed = tf.io.parse_single_example(serialized=serial_exmp,
                                                        features=context_features)
            sequence_id = [
                tf.sparse_to_dense(
                    context_parsed['sequence_id_' + str(i)].indices,
                    [config['maxlen']],
                    context_parsed['sequence_id_' + str(i)].values
                ) for i in range(config['seq_num'])
            ]
            dense_feature = tf.sparse.to_dense(context_parsed['dense_feature'])
            category_feature = tf.sparse.to_dense(context_parsed['category_feature'])
            slate_label = tf.sparse.to_dense(context_parsed['slate_label'])
            label = context_parsed['label']

            return sequence_id, dense_feature, category_feature, slate_label, label

        def _flat_map_fn(sequence_id,
                         dense_feature,
                         category_feature,
                         slate_label,
                         label):
            batch_size = self.batch_size
            return dataset_ops.Dataset.zip((
                sequence_id.padded_batch(
                    batch_size,
                    padded_shapes=([config['seq_num'], config['maxlen']])
                ),
                dense_feature.padded_batch(
                    batch_size=batch_size,
                    padded_shapes=([config['dense_feature_num']])
                ),
                category_feature.padded_batch(
                    batch_size,
                    padded_shapes=([config['category_feature_num']])
                ),
                slate_label.batch(batch_size=batch_size),
                label.batch(batch_size=batch_size),
            ))

        def preprocess_fn(*args):
            """A transformation function to preprocess raw data
            into trainable input.
            """
            return args[:-1], tf.one_hot(args[-1], self.class_num)

        def preprocess_slate_fn(*args):
            """A transformation function to preprocess raw data
            into trainable input.
            """
            return args[:-1], args[-2]

        # tf.enable_eager_execution()
        if is_slate_label:
            dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=4)
            if is_pred:
                dataset_iter = dataset \
                    .map(_parse_exmp, num_parallel_calls=1) \
                    .window(size=self.batch_size, drop_remainder=False) \
                    .flat_map(_flat_map_fn) \
                    .map(preprocess_slate_fn)
            else:
                dataset_iter = dataset \
                    .map(_parse_exmp, num_parallel_calls=4) \
                    .shuffle(10000) \
                    .window(size=self.batch_size, drop_remainder=True) \
                    .flat_map(_flat_map_fn) \
                    .map(preprocess_slate_fn) \
                    .repeat()
        else:
            dataset = tf.data.TFRecordDataset(filename, num_parallel_reads=4)
            if is_pred:
                dataset_iter = dataset \
                    .map(_parse_exmp, num_parallel_calls=1) \
                    .window(size=self.batch_size, drop_remainder=False) \
                    .flat_map(_flat_map_fn) \
                    .map(preprocess_fn)
            else:
                dataset_iter = dataset \
                    .map(_parse_exmp, num_parallel_calls=4) \
                    .shuffle(10000) \
                    .window(size=self.batch_size, drop_remainder=True) \
                    .flat_map(_flat_map_fn) \
                    .map(preprocess_fn) \
                    .repeat()
        return dataset_iter

    def to_tfrecord(self, data, filename):
        """Extract features from the original data in csv format, and convert to tfrecord format.

        Note:
            Tensorflow uses tensorflow.TFRecordDataloader to load tfrecord files

            Pytorch uses pytorch.TFRecordDataloader to load tfrecord files

        Args:
            csv_file (str): the file, in which original csv data is saved
            h5_file (str): the file, to which tfrecord data will save

        Returns:

        """

        writer = tf.python_io.TFRecordWriter(filename)
        for jj in range(0, len(data), 10000):
            (sequence_ids, dense_features, category_features_ids, slate_labels), labels \
                = self.feature_extraction(data[jj: min(jj + 10000, len(data))])
            for i in range(len(sequence_ids)):
                sequence_id = sequence_ids[i]
                dense_feature = dense_features[i]
                category_features_id = category_features_ids[i]
                slate_label = slate_labels[i]
                label = labels[i]

                sequence_id_feature = [
                    tf.train.Feature(int64_list=tf.train.Int64List(value=sequence_id[i]))
                    for i in range(self.seq_num)
                ]
                dense_feature_feature = tf.train.Feature(
                    float_list=tf.train.FloatList(value=dense_feature)
                )
                category_features_id_feature = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=category_features_id)
                )
                slate_label = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=slate_label)
                )
                label_feature = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label])
                )

                feature = {
                    "dense_feature": dense_feature_feature,
                    "category_feature": category_features_id_feature,
                    "slate_label": slate_label,
                    "label": label_feature,
                }

                for seq_num_i in range(self.seq_num):
                    feature['sequence_id_' + str(seq_num_i)] = sequence_id_feature[seq_num_i]

                seq_example = tf.train.Example(
                    features=tf.train.Features(feature=feature)
                )
                writer.write(seq_example.SerializeToString())
        writer.close()
