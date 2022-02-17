import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.util import nest
from .utils import index_matrix_to_pairs_fn

try:
    from tensorflow.contrib.layers.python.layers import utils  # 1.0.0
except:
    from tensorflow.contrib.layers import utils

smart_cond = utils.smart_cond

try:
    LSTMCell = rnn.LSTMCell  # 1.0.0
    MultiRNNCell = rnn.MultiRNNCell
except:
    LSTMCell = tf.contrib.rnn.LSTMCell
    MultiRNNCell = tf.contrib.rnn.MultiRNNCell


def ptn_rnn_decoder(cell,
                    decoder_target_ids,
                    enc_outputs,
                    enc_final_states,
                    seq_length,
                    res_length,
                    hidden_dim,
                    num_glimpse,
                    batch_size,
                    l1_mask,
                    l2_mask,
                    l3_mask,
                    l0_ssr_mask,
                    initializer=None,
                    mode="SAMPLE",
                    reuse=False,
                    beam_size=None,
                    temperature=1,
                    train_sample='random',
                    predict_sample='random'):
    """
    :param cell:
    :param decoder_target_ids:
    :param enc_outputs:
    :param enc_final_states:
    :param seq_length:
    :param hidden_dim:
    :param num_glimpse:
    :param batch_size:
    :param initializer:
    :param mode: SAMPLE/GREEDY/BEAMSEARCH/TRAIN, if TRAIN, decoder_input_ids shouldn't be none
    :param reuse:
    :param beam_size: a positive int if mode="BEAMSEARCH"
    :return: [logits, sampled_ids, final_state], shape: [batch_size, seq_len, data_len], [batch, seq_len], state_size
    """
    with tf.variable_scope("decoder_rnn") as scope:
        if reuse:
            scope.reuse_variables()

        first_decoder_input = trainable_initial_state(
            batch_size, hidden_dim, initializer=None, name="first_decoder_input")

        enc_refs = {}
        output_ref = []
        index_matrix_to_pairs = index_matrix_to_pairs_fn(batch_size, seq_length)

        def intra_attention(bef, query, scope="intra_attention"):
            with tf.variable_scope(scope) as scope:
                W_b = tf.get_variable(
                    "W_b", [hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                v_dec = tf.get_variable(
                    "v_dec", [hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                W_bef = tf.get_variable(
                    "W_bef", [1, hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                bias_dec = tf.get_variable(
                    "bias_dec", [hidden_dim],
                    initializer=tf.zeros_initializer)
                if len(bef) <= 1:
                    if len(bef) == 0:
                        return tf.zeros([batch_size, hidden_dim])
                    else:
                        return bef[0]
                else:
                    bef = tf.stack(bef, axis=1)
                    # bef_rs = tf.reduce_sum(bef_s,axis=[2])

                decoded_bef = tf.nn.conv1d(bef, W_bef, 1, "VALID",
                                           name="decoded_bef")  # [batch, decoder_len, hidden_dim]
                decoded_query = tf.expand_dims(tf.matmul(query, W_b, name="decoded_query"), 1)  # [batch, 1, hidden_dim]
                scores = tf.reduce_sum(v_dec * tf.tanh(decoded_bef + decoded_query + bias_dec),
                                       [-1])  # [batch, decoder_len]
                p1 = tf.nn.softmax(scores)
                aligments1 = tf.expand_dims(p1, axis=2)
                return tf.reduce_sum(aligments1 * bef, axis=[1])

        def attention(ref, query, dec_ref, with_softmax, scope="attention"):
            with tf.variable_scope(scope) as scope:
                W_q = tf.get_variable(
                    "W_q", [hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                W_dec = tf.get_variable(
                    "W_dec", [hidden_dim, hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                v = tf.get_variable(
                    "v", [hidden_dim],
                    initializer=tf.contrib.layers.xavier_initializer())
                bias = tf.get_variable(
                    "bias", [hidden_dim],
                    initializer=tf.zeros_initializer)

                enc_ref_key = (ref.name, scope.name)
                if enc_ref_key not in enc_refs:
                    W_ref = tf.get_variable("W_ref", [1, hidden_dim, hidden_dim],
                                            initializer=tf.contrib.layers.xavier_initializer())
                    enc_refs[enc_ref_key] = tf.nn.conv1d(ref, W_ref, 1, "VALID",
                                                         name="encoded_ref")  # [batch, data_len, hidden_dim]
                encoded_ref = enc_refs[enc_ref_key]
                encoded_query = tf.expand_dims(tf.matmul(query, W_q, name="encoded_query"), 1)  # [batch, 1, hidden_dim]
                decoded_ref = tf.expand_dims(tf.matmul(dec_ref, W_dec, name="decoded_ref"), 1)  # [batch, 1, hidden_dim]
                scores = tf.reduce_sum(v * tf.tanh(encoded_ref + encoded_query + decoded_ref + bias),
                                       [-1])  # [batch, data_len]

                if with_softmax:
                    return tf.nn.softmax(scores)
                else:
                    return scores

        def glimpse(ref, query, dec_ref, scope="glimpse"):
            p = attention(ref, query, dec_ref, with_softmax=True, scope=scope)
            alignments = tf.expand_dims(p, axis=2)  # [batch, data_len, 1]
            return tf.reduce_sum(alignments * ref, axis=[1])

        def output_fn(ref, query, dec_ref, num_glimpse):
            for idx in range(num_glimpse):
                query = glimpse(ref, query, dec_ref, "glimpse_{}".format(idx))
            return attention(ref, query, dec_ref, with_softmax=False, scope="attention")

        def input_fn(input_idx):
            input_index_pairs = tf.stop_gradient(index_matrix_to_pairs(input_idx))
            return tf.gather_nd(enc_outputs, input_index_pairs)

        def random_sample_from_logits(logits):
            logits = logits / temperature
            sampled_idx = tf.cast(tf.multinomial(logits=logits, num_samples=1), dtype='int32')  # [batch_size,1]
            sampled_idx = tf.reshape(sampled_idx, [batch_size])  # [batch_size]
            return sampled_idx

        def greedy_sample_from_logits(logits):
            # logits: [batch, seq_length]
            return tf.cast(tf.argmax(logits, 1), tf.int32)

        def call_cell(input_idx, state, point_mask):
            """
              call lstm_cell and compute attention and intra-attention
            :param input_idx: [batch]
            :param state:
            :param point_mask: [batch, seq_length]
            :return: [batch_size, seq_length]
            """
            if input_idx is not None:
                _input = input_fn(input_idx)  # [batch_size, hidden_dim]
            else:
                _input = first_decoder_input

            cell_output, new_state = cell(_input, state)
            intra_dec = intra_attention(output_ref, cell_output)  # [batch_size, hidden_dim]
            output_ref.append(cell_output)
            logits = output_fn(enc_outputs, cell_output, intra_dec, num_glimpse)  # [batch_size, data_len]
            if point_mask is not None:
                paddings = tf.ones_like(logits) * (-2 ** 32 + 1)
                logits = tf.where(tf.equal(point_mask, True), logits, paddings)
            return logits, new_state

        selected_mask = tf.cast(tf.ones([batch_size, seq_length], dtype=tf.int32), tf.bool)
        l_mask = l1_mask
        point_mask = tf.math.logical_and(selected_mask, l_mask)

        logits, state = call_cell(input_idx=None, state=enc_final_states, point_mask=point_mask)  # [batch_size, data_len]
        scope.reuse_variables()
        output_logits = [logits]
        if (mode in ['SAMPLE', "GREEDY", "CUSTOM"]):
            if mode == "SAMPLE":
                sample_fn = random_sample_from_logits
            elif mode == "GREEDY":
                sample_fn = greedy_sample_from_logits
            elif mode == 'CUSTOM':
                sample_fn = random_sample_from_logits
            else:
                raise NotImplementedError("invalid mode: %s. Available modes: [SAMPLE, GREEDY, CUSTOM]" % mode)

            output_idx = sample_fn(logits)  # [batch_size]
            output_idxs = [output_idx]

            for i in range(1, res_length):
                onehot = tf.one_hot(output_idx, depth=seq_length, on_value=False, off_value=True, dtype=tf.bool)
                selected_mask = tf.math.logical_and(selected_mask, onehot)

                if i == 1:
                    l_mask = l1_mask
                elif i == 2:
                    l_mask = l1_mask
                elif i == 3:
                    l_mask = l2_mask
                elif i == 4:
                    l_mask = l2_mask
                elif i == 5:
                    l_mask = l2_mask
                elif i == 6:
                    l_mask = l3_mask
                elif i == 7:
                    l_mask = l3_mask
                elif i == 8:
                    l_mask = l3_mask
                else:
                    raise Exception('update point_mask failed, unexcepted i')

                point_mask = tf.math.logical_and(l_mask, selected_mask)
                ssr_mask = tf.math.logical_not(tf.math.logical_and(tf.reduce_any(tf.math.logical_and(l0_ssr_mask, tf.math.logical_not(selected_mask)), axis=1, keep_dims=True), l0_ssr_mask))
                point_mask = tf.math.logical_and(point_mask, ssr_mask)
                logits, state = call_cell(output_idx, state, point_mask)  # [batch_size, data_len]
                output_logits.append(logits)
                output_idx = sample_fn(logits)  # [batch_size]
                output_idxs.append(output_idx)

            return tf.stack(output_logits, axis=1), tf.stack(output_idxs, axis=1), state
        elif mode == "TRAIN":
            output_idxs = tf.unstack(decoder_target_ids, axis=1)
            output_idx = output_idxs[0]  # [batch_size]

            for i in range(1, res_length):
                onehot = tf.one_hot(output_idx, depth=seq_length, on_value=False, off_value=True, dtype=tf.bool)
                selected_mask = tf.math.logical_and(selected_mask, onehot)

                if i == 1:
                    l_mask = l1_mask
                elif i == 2:
                    l_mask = l1_mask
                elif i == 3:
                    l_mask = l2_mask
                elif i == 4:
                    l_mask = l2_mask
                elif i == 5:
                    l_mask = l2_mask
                elif i == 6:
                    l_mask = l3_mask
                elif i == 7:
                    l_mask = l3_mask
                elif i == 8:
                    l_mask = l3_mask
                else:
                    raise Exception('update point_mask failed, unexcepted i')

                point_mask = tf.math.logical_and(l_mask, selected_mask)

                ssr_mask = tf.math.logical_not(tf.math.logical_and(tf.reduce_any(tf.math.logical_and(l0_ssr_mask, tf.math.logical_not(selected_mask)), axis=1, keep_dims=True), l0_ssr_mask))
                point_mask = tf.math.logical_and(ssr_mask, point_mask)

                logits, state = call_cell(output_idx, state, point_mask)  # [batch_size, data_len]
                output_logits.append(logits)
                output_idx = output_idxs[i]  # [batch_size]

            return tf.stack(output_logits, axis=1), state
        elif mode == "BEAMSEARCH":

            index_matrix_to_beampairs = index_matrix_to_pairs_fn(batch_size, beam_size)

            def top_k(acum_logits, logits):
                """
                :param acum_logits: [batch] * beam_size
                :param logits: [batch, len] * beam_size
                :return:
                  new_acum_logits [batch] * beam_size,
                  last_beam_id [batch, beam_size], sample_id [batch, beam_size]
                """
                candicate_size = len(logits)
                local_acum_logits = logits
                if accum_logits is not None:
                    local_acum_logits = [tf.reshape(acum_logits[ik], [-1, 1]) + logits[ik]
                                         for ik in range(candicate_size)]
                local_acum_logits = tf.concat(local_acum_logits, axis=1)
                local_acum_logits, local_id = tf.nn.top_k(local_acum_logits, beam_size)
                last_beam_id = local_id // seq_length
                last_beam_id = index_matrix_to_beampairs(last_beam_id)  # [batch, beam_size, 2]
                sample_id = local_id % seq_length
                new_acum_logits = tf.unstack(local_acum_logits, axis=1)  # [batch] * beam_size
                return new_acum_logits, last_beam_id, sample_id

            def beam_select(inputs_l, beam_id):
                """
                :param input_l: list of tensors, len(input_l) = k
                :param beam_id: [batch, k, 2]
                :return: output_l, list of tensors, len = k
                """

                def _select(input_l):
                    input_l = tf.stack(input_l, axis=1)  # [batch, beam_size, ...]
                    output_l = tf.gather_nd(input_l, beam_id)  # [batch, beam_size, ...]
                    output_l = tf.unstack(output_l, axis=1)
                    return output_l

                # [state, state] -> [(h,c),(h,c)] -> [[h,h,h], [c,c,c]]
                inputs_ta_flat = zip(*[nest.flatten(input_l) for input_l in inputs_l])
                # [[h,h,h], [c,c,c]] -(beam select)> [[h,h,h], [c,c,c]]
                outputs_ta_flat = [_select(input_ta) for input_ta in inputs_ta_flat]
                # [[h,h,h], [c,c,c]] -> [(h,c),(h,c)] -> [state, state]
                a = [output_ta_flat for output_ta_flat in list(zip(*outputs_ta_flat))]
                outputs_l = [nest.pack_sequence_as(inputs_l[0], output_ta_flat)
                             for output_ta_flat in list(zip(*outputs_ta_flat))]
                return outputs_l

            def beam_sample(accum_logits, logits, point_mask, selected_mask, state, pre_output_idxs, i):

                # sample top_k, last_bema_id:[batch,beam_size], output_idx:[batch,beam_size]
                accum_logits, last_beam_id, output_idx = top_k(accum_logits,
                                                               logits)  # [batch, beam_size], 前面那个beam path, 后面哪个节点
                state = beam_select(state, last_beam_id)
                point_mask = beam_select(point_mask, last_beam_id)
                selected_mask = beam_select(selected_mask, last_beam_id)
                output_idx = tf.unstack(output_idx, axis=1)  # [batch] * beam_size

                selected_mask = [
                    tf.math.logical_and(selected_mask[j],
                                        tf.one_hot(output_idx[j], depth=seq_length, on_value=False, off_value=True, dtype=tf.bool))
                    for j in range(beam_size)]

                l_output_idx = [tf.expand_dims(t, axis=1)  # [batch, 1] * beam_size
                                for t in output_idx]
                if pre_output_idxs is not None:
                    pre_output_idxs = beam_select(pre_output_idxs, last_beam_id)
                    output_idxs = list(map(lambda ts: tf.concat(ts, axis=1), zip(pre_output_idxs, l_output_idx)))
                else:
                    output_idxs = l_output_idx
                return accum_logits, point_mask, selected_mask, state, output_idx, output_idxs

            # initial setting
            state = [state] * beam_size  # [batch, state_size] * beam_size
            point_mask = [point_mask] * beam_size  # [batch, data_len] * beam_size
            selected_mask = [selected_mask] * beam_size
            # logits -> log pi
            logits = logits - tf.reduce_logsumexp(logits, axis=1, keep_dims=True)
            logits = [logits] * beam_size  # [batch, data_len] * beam_size
            accum_logits = [tf.zeros([batch_size])] * beam_size

            accum_logits, point_mask, selected_mask, state, output_idx, output_idxs = \
                beam_sample(accum_logits, logits, point_mask, selected_mask, state, None, 0)
            for i in range(1, res_length):
                point_mask_new = []
                for j in range(beam_size):
                    selected_mask_ = selected_mask[j]

                    if i == 1:
                        l_mask = l1_mask
                    elif i == 2:
                        l_mask = l1_mask
                    elif i == 3:
                        l_mask = l2_mask
                    elif i == 4:
                        l_mask = l2_mask
                    elif i == 5:
                        l_mask = l2_mask
                    elif i == 6:
                        l_mask = l3_mask
                    elif i == 7:
                        l_mask = l3_mask
                    elif i == 8:
                        l_mask = l3_mask
                    else:
                        raise Exception('update point_mask failed, unexcepted i')

                    point_mask_ = tf.math.logical_and(l_mask, selected_mask_)
                    ssr_mask = tf.math.logical_not(tf.math.logical_and(tf.reduce_any(tf.math.logical_and(l0_ssr_mask, tf.math.logical_not(selected_mask_)), axis=1, keep_dims=True), l0_ssr_mask))
                    point_mask_ = tf.math.logical_and(point_mask_, ssr_mask)
                    point_mask_new.append(point_mask_)
                point_mask = point_mask_new

                logits, state = zip(*[call_cell(output_idx[ik], state[ik], point_mask[ik])  # [batch_size, data_len]
                                      for ik in range(beam_size)])
                # logits -> log pi
                logits = [logit_ - tf.reduce_logsumexp(logit_, axis=1, keep_dims=True) for logit_ in logits]
                accum_logits, point_mask, selected_mask, state, output_idx, output_idxs = \
                    beam_sample(accum_logits, logits, point_mask, selected_mask, state, output_idxs, i)
            return accum_logits[0], output_idxs[0], state[0]
        else:
            raise NotImplementedError("unknown mode: %s. Available modes: [SAMPLE, TRAIN, GREEDY, BEAMSEARCH]" % mode)


def trainable_initial_state(batch_size,
                            state_size,
                            initializer=None,
                            name="initial_state"):
    flat_state_size = nest.flatten(state_size)  # Returns a flat sequence from a given nested structure.

    if not initializer:
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
    else:
        flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

    names = ["{}_{}".format(name, i) for i in range(len(flat_state_size))]
    tiled_states = []

    for name, size, init in zip(names, flat_state_size, flat_initializer):
        shape_with_batch_dim = [1, size]
        initial_state_variable = tf.get_variable(
            name, shape=shape_with_batch_dim, initializer=init())

        tiled_state = tf.tile(initial_state_variable,
                              [batch_size, 1], name=(name + "_tiled"))
        tiled_states.append(tiled_state)

    return nest.pack_sequence_as(structure=state_size,
                                 flat_sequence=tiled_states)


def ctr_dicriminator(user, card, hidden_dim, res_length=500):
    '''
    :param user: [batch_size, user_embedding]
    :param card: [batch_size, res_len, item_embedding]
    :param hidden_dim: dnn hidden dimension
    :return: logit for ctr
    '''
    with tf.variable_scope("ctr_dicriminator"):
        batch_size = user.get_shape()[0].value
        if batch_size is None:
            batch_size = tf.shape(user)[0]

        user_flat = tf.stack(res_length * [user], axis=1)

        cross_feature = tf.reduce_sum(tf.multiply(user_flat, card), axis=2)
        cross_feature = tf.reshape(cross_feature, shape=[batch_size, -1])

        card_feature = tf.reshape(card, shape=[batch_size, -1])

        feature = tf.concat([user, card_feature, cross_feature], axis=1)
        feature = tf.layers.dense(feature, hidden_dim, activation=tf.nn.relu)
        logits = tf.layers.dense(feature, 1, activation=None)
        logits = tf.squeeze(logits, axis=[1])
        return logits
