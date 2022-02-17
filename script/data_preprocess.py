from rl4rs.utils.datautil import FeatureUtil
import numpy as np
import sys, os, random


def data_augment(file, out_file):
    f = open(out_file, 'w')
    data = open(file, 'r').read().split('\n')
    data_size = len(data)
    print('data length', data_size)
    tmp = []
    role_id_prev = None
    for record in data:
        if len(record) < 1 or 'timestamp' in record:
            continue
        role_id = record.split('@')[1]
        if role_id == role_id_prev or role_id_prev is None:
            tmp.append(record)
            role_id_prev = role_id
        else:
            assert len(tmp) <= 4
            for i in range(len(tmp), 4):
                timestamp, session_id, sequence_id, exposed_items, user_feedback, \
                    user_seqfeature, user_protrait, item_feature, behavior_policy_id = tmp[-1].split('@')
                timestamp_new = str(int(timestamp) + 1)
                sequence_id_new = str(int(sequence_id) + 1)
                random_i = np.random.randint(1, data_size - 1)
                exposed_items_new = data[random_i].split('@')[3]
                item_feature_new = data[random_i].split('@')[7]
                user_feedback_new = '0,0,0,0,0,0,0,0,0'
                tmp.append('@'.join([
                    timestamp_new,
                    session_id,
                    sequence_id_new,
                    exposed_items_new,
                    user_feedback_new,
                    user_seqfeature,
                    user_protrait,
                    item_feature_new,
                    behavior_policy_id
                ]))
            print(*tmp, sep='\n', end='\n', file=f)
            tmp = [record]
            role_id_prev = role_id
    f.close()


def slate2trajectory(file, out_file):
    f = open(out_file, 'w')
    data = open(file, 'r').read().split('\n')
    data_size = len(data)
    print('data length', data_size)
    tmp = []
    role_id_prev = None
    for record in data:
        if len(record) < 1 or 'timestamp' in record:
            continue
        role_id = record.split('@')[1]
        if role_id == role_id_prev or role_id_prev is None:
            tmp.append(record)
            role_id_prev = role_id
        else:
            assert len(tmp) == 4
            # timestamp, session_id, sequence_id, exposed_items, user_feedback, user_seqfeature, user_protrait, item_feature, behavior_policy_id
            timestamp = tmp[0].split('@')[0]
            session_id = tmp[0].split('@')[1]
            sequence_id = '1'
            exposed_items = ','.join([x.split('@')[3] for x in tmp])
            user_feedback = ','.join([x.split('@')[4] for x in tmp])
            user_seqfeature = tmp[0].split('@')[5]
            user_protrait = tmp[0].split('@')[6]
            item_feature = ';'.join([x.split('@')[7] for x in tmp])
            behavior_policy_id = tmp[0].split('@')[8]
            traj = [
                timestamp,
                session_id,
                sequence_id,
                exposed_items,
                user_feedback,
                user_seqfeature,
                user_protrait,
                item_feature,
                behavior_policy_id
            ]
            print(*traj, sep='@', end='\n', file=f)
            tmp = [record]
            role_id_prev = role_id
    f.close()


def dataset2tfrecord(config, file, tfrecord_file, is_slate):
    def feature_construct(session, is_slate):
        samples = []
        for i in range(len(session)):
            _, _, sequence_id, exposed_items, user_feedback, user_seqfeature, \
            user_protrait, item_feature, _ = FeatureUtil.record_split(session[i])
            assert sequence_id - 1 == i
            user_protrait_category = user_protrait[:10]
            user_protrait_dense = user_protrait[10:]
            category_feature = user_protrait_category + [sequence_id] + exposed_items
            prev_items = [session[ii].split('@')[3].split(',')[jj] for ii in range(i) for jj in range(9)]
            prev_items = list(map(int, prev_items))
            sequence_feature_clicked = prev_items if i > 0 else [0]
            sequence_feature = [user_seqfeature, sequence_feature_clicked]
            if is_slate:
                # label = '0'
                label = 0
                samples.append((
                    role_id_prev,
                    sequence_feature,
                    user_protrait_dense + item_feature,
                    category_feature,
                    user_feedback,
                    label
                ))
            else:
                for j in range(9):
                    item_id = exposed_items[j]
                    label = user_feedback[j]
                    item_feature_size = len(item_feature) // 9
                    item_feature_j = item_feature[item_feature_size * j:item_feature_size * (j + 1)]
                    category_feature_j = category_feature + [item_id]
                    dense_feature_j = item_feature + item_feature_j
                    samples.append((
                        role_id_prev,
                        sequence_feature,
                        user_protrait_dense + dense_feature_j,
                        category_feature_j,
                        user_feedback,
                        label
                    ))
        return samples
    featureutil = FeatureUtil(config)
    data = open(file, 'r').read().split('\n')
    print('data length', len(data))
    # role_id, sequence_feature, dense_feature, category_feature, label
    # timestamp@session_id@sequence_id@exposed_items@user_feedback@user_seqfeature@user_protrait@item_feature@behavior_policy_id
    tmp = []
    records = []
    role_id_prev = None
    for record in data:
        if len(record) < 1 or 'timestamp' in record:
            continue
        role_id = record.split('@')[1]
        if role_id == role_id_prev or role_id_prev is None:
            tmp.append(record)
            role_id_prev = role_id
        else:
            samples = feature_construct(tmp, is_slate)
            records = records + samples
            tmp = [record]
            role_id_prev = role_id
    if len(tmp) > 0:
        samples = feature_construct(tmp, is_slate)
        records = records + samples
    print('tfrecord length', len(records), records[0])
    random.shuffle(records)
    featureutil.to_tfrecord(records, tfrecord_file)


config = {
    "maxlen": 64,
    "batch_size": 32,
    "class_num": 2,
    "dense_feature_num": 432,
    "category_feature_num": 21,
    "category_hash_size": 100000,
    "seq_num": 2
}
file = sys.argv[1]
out_file = sys.argv[2]
stage = sys.argv[3]
assert stage in ('data_augment', 'slate2trajectory', 'tfrecord_item', 'tfrecord_slate')
if stage == 'data_augment':
    data_augment(file, out_file)
if stage == 'slate2trajectory':
    slate2trajectory(file, out_file)
if stage == 'tfrecord_item':
    dataset2tfrecord(config, file, out_file, is_slate=False)
if stage == 'tfrecord_slate':
    dataset2tfrecord(config, file, out_file, is_slate=True)
