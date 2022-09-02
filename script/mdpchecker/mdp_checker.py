import numpy as np
import sys
import random
from scipy.stats import spearmanr
from keras_transformer import get_model, decode
from rl4rs.mdpchecker.decoder import beam_search, token_probs

# dataset_file = 'recsys15.csv'
# dataset_file = 'movielens.csv'
# dataset_file = 'rl4rs.csv'
# dataset_file = 'lastfm.csv'
# dataset_file = 'cikm2016.csv'
dataset_file = sys.argv[1] + '.csv'
dataset_dir = sys.argv[2]

# the data of recsys15 relative to the
# number of commodities is too sparse,
# increase the sample size
if 'recsys15' in dataset_file:
    source_len = 8
elif 'cikm2016' in dataset_file:
    source_len = 5
else:
    source_len = 16
target_len = 5
np.random.seed(1)

data = open(dataset_dir + '/' + dataset_file).read().split('\n')[1:-1]

source_tokens = []
target_tokens = []
for sample in data:
    user_id, items = sample.split(' ')
    item_list = items.split(',')
    if len(item_list) >= source_len + target_len:
        # assert len(item_list) >= source_len + target_len
        i = 0
        if 'rl4rs' in dataset_file or 'cikm2016' in dataset_file:
            source_tokens.append(item_list[:source_len])
            target_tokens.append(item_list[source_len:source_len + target_len])
        else:
            while i + source_len + target_len < len(item_list):
                source_tokens.append(item_list[i: i + source_len])
                target_tokens.append(item_list[i + source_len: i + source_len + target_len])
                i = i + np.random.randint(source_len, source_len + target_len) // 6
    else:
        print('len(item_list) <= source_len + target_len in', '\t',sample)

# Generate dictionaries
token_dict = {
    '<PAD>': 0,
    '<START>': 1,
    '<END>': 2,
}


def build_token_dict(token_list):
    for tokens in token_list:
        for token in tokens:
            if token not in token_dict:
                token_dict[token] = len(token_dict)
    return token_dict


source_token_dict = build_token_dict(source_tokens)
target_token_dict = build_token_dict(target_tokens)
target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

# Add special tokens
encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]

# Padding
source_max_len = max(map(len, encode_tokens))
target_max_len = max(map(len, decode_tokens))

encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]

encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]

print('sample lens:', len(encode_input))
print('source_token_dict lens:', len(source_token_dict))
print('target_token_dict lens:', len(target_token_dict))
# [1, 3, 4, 5, 6, 2] [1, 3, 4, 5, 6, 7, 8, 9, 2] [[3], [4], [5], [6], [7], [8], [9], [2], [0]]
# print(encode_input[0], decode_input[0], decode_output[0])

# Build & fit model
model = get_model(
    token_num=max(len(source_token_dict), len(target_token_dict)),
    embed_dim=256,
    encoder_num=1,
    decoder_num=1,
    head_num=1,
    hidden_dim=128,
    dropout_rate=0.05,
    use_same_embed=False,  # Use different embeddings for different languages
)

model.compile('adam', 'sparse_categorical_crossentropy')
model.summary()

model.fit(
    x=[np.array(encode_input)[:-10000], np.array(decode_input)[:-10000]],
    y=np.array(decode_output)[:-10000],
    epochs=20,
    batch_size=256,
    shuffle=True,
    verbose=2
)

model.save_weights(dataset_file.split('.')[0] + '.h5')

# Load
model.load_weights(dataset_file.split('.')[0] + '.h5')

# greedy result print & input output comparison
# decoded = decode(
#     model,
#     encode_input[:1024],
#     start_token=target_token_dict['<START>'],
#     end_token=target_token_dict['<END>'],
#     pad_token=target_token_dict['<PAD>'],
#     top_k=1
# )
# print([target_token_dict_inv[x] for x in decode_input[0]], [target_token_dict_inv[x] for x in decoded[0]])
# print([target_token_dict_inv[x] for x in decode_input[1]], [target_token_dict_inv[x] for x in decoded[1]])

# beam search
batch_size = 2048
beam_size = 100
# use 20 hot items since rl4rs has only 200+ items
hot_beam_size = 20 if 'rl4rs' in dataset_file else beam_size
# cikm2016 has only 60853 items
candidates_size = 6000 if 'cikm2016' in dataset_file else hot_beam_size
random.seed(1)
encode_input = random.sample(encode_input[-10000:], batch_size)
output_greedy, greedy_score = beam_search(model, encode_input, beam_size=1, target_len=target_len)
output_topk, beam_score = beam_search(model, encode_input, beam_size=beam_size, target_len=target_len)
# np.savez(dataset_file.split('.')[0]+'.npz', output_topk=output_topk, beam_score=beam_score)
# npzdata = np.load(dataset_file.split('.')[0] + '.npz')
# output_topk = npzdata['output_topk']
# beam_score = npzdata['beam_score']

output_topk_5, beam_score_5 = output_topk[:, :int(beam_size * 0.05)], beam_score[:, :int(beam_size * 0.05)]
output_topk_20, beam_score_20 = output_topk[:, :int(beam_size * 0.2)], beam_score[:, :int(beam_size * 0.2)]
output_topk_hot, beam_score_hot = beam_search(model, encode_input, beam_size=hot_beam_size, target_len=target_len, use_candidates=True, candidates_size=candidates_size)
output_topk_hot5, beam_score_hot5 = output_topk_hot[:, :int(beam_size * 0.05)], beam_score_hot[:, :int(beam_size * 0.05)]
output_topk_hot20, beam_score_hot20 = output_topk_hot[:, :int(beam_size * 0.2)], beam_score_hot[:, :int(beam_size * 0.2)]

greedy_score = np.nanmean(greedy_score, axis=1)
top_5_percent_score = np.nanmean(beam_score_5, axis=1)
top_20_percent_score = np.nanmean(beam_score_20, axis=1)
hot_5_percent_score = np.nanmean(beam_score_hot5, axis=1)
hot_20_percent_score = np.nanmean(beam_score_hot20, axis=1)

print('experiment II results')
print('top_5_percent_score top_20_percent_score greedy_score hot_5_percent_score hot_20_percent_score')
print(1,
      np.nanmean(top_20_percent_score / top_5_percent_score),
      np.nanmean(greedy_score / top_5_percent_score),
      np.nanmean(hot_5_percent_score / top_5_percent_score),
      np.nanmean(hot_20_percent_score / top_5_percent_score))

print('experiment I start')
tmp = []
for j in range(int(beam_size)):
    batch_outputs = output_topk[:, j]
    probs = []
    for i in range(5):
        prob = token_probs(model, encode_input, batch_outputs[:, :i + 1])[list(range(batch_size)), output_topk[:, j, i + 1]]
        probs.append(prob)
    tmp.append(probs)
probs = np.array(tmp).swapaxes(0, 2).swapaxes(1, 2)
metrics = []
for j in range(batch_size):
    prob = probs[j]
    prob_sum = np.sum(prob, axis=1)
    seq_score = np.multiply.reduce(np.array(prob), axis=1)
    for i in range(5):
        metrics.append((np.corrcoef(np.multiply.reduce(np.array(prob[:, :i + 1]), axis=1), seq_score)[0][1],
                        spearmanr(np.multiply.reduce(np.array(prob[:, :i + 1]), axis=1), seq_score)[0]))
metrics = np.array(metrics).reshape((batch_size, 5, 2))
metrics = np.nan_to_num(metrics, nan=1.0)
print('experiment I results')
print('corrcoef', ' ', 'spearman')
print(np.nanmean(metrics, axis=0))