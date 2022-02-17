import os
import sys
import glob
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
if tf.test.is_gpu_available():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from tensorflow import keras
from rl4rs.utils.datautil import FeatureUtil
from rl4rs.utils.fileutil import find_match_files

config = {
    "epoch": 20,
    "maxlen": 64,
    "batch_size": 256,
    "class_num": 2,
    "dense_feature_num": 432,
    "category_feature_num": 21,
    "category_hash_size": 100000,
    "seq_num": 2,
    "emb_size": 128,
    "hidden_units": 128,
    "action_size": 284
}
train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
model_type = sys.argv[4]
is_slate_label = bool(int(sys.argv[5]))
featureutil = FeatureUtil(config)

train_files = [match for match in find_match_files(train_file + '*', train_file)]
test_files = [match for match in find_match_files(test_file + '*', test_file)]
print('train on ', train_files, ' test on ', test_files)
iter_train = featureutil.read_tfrecord(train_files, is_slate_label=is_slate_label)
iter_test = featureutil.read_tfrecord(test_files, is_slate_label=is_slate_label)
model = __import__("rl4rs.nets." + model_type, fromlist=['get_model']).get_model(config)
steps_per_epoch = 600000 // config["batch_size"]
steps_per_epoch_val = 400000 // config["batch_size"]
earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='min')
model.fit(iter_train, steps_per_epoch=steps_per_epoch, epochs=int(config["epoch"]),
          validation_data=iter_test, validation_steps=steps_per_epoch_val, verbose=2, callbacks=[earlyStopping])

saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
saver.save(sess, model_file)
