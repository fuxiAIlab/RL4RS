import numpy as np


def token_probs(model,
                batch_inputs,
                batch_outputs):
    return np.array(model.predict([np.array(batch_inputs), np.array(batch_outputs)]))[:, -1]


def decode_step(model,
                batch_inputs,
                batch_outputs,
                candidates=None,
                beam_size=1):
    predicts = model.predict([np.array(batch_inputs), np.array(batch_outputs)])
    tmp = []
    for i in range(len(predicts)):
        probs = [(prob, j) for j, prob in enumerate(predicts[i][-1])]
        if candidates is not None:
            probs = [x if x[1] in candidates[i] else (0, x[1]) for x in probs]
        probs.sort(reverse=True)
        probs = probs[:beam_size]
        tmp.append(probs)
    return np.array(tmp)


def beam_search(model, encode_input, beam_size, target_len, use_candidates=False):
    batch_size = len(encode_input)
    output_topk = np.zeros((batch_size, beam_size, target_len + 1), dtype=np.int)
    beam_score = np.ones((batch_size, beam_size))
    output_topk[:, :, 0] = 1
    # probs = []
    candidates = None
    prob = decode_step(model, encode_input, output_topk[:, 0, :1], candidates=candidates, beam_size=beam_size)
    if use_candidates:
        candidates = prob[:, :, 1]
    output_topk[:, :, 1] = prob[:, :, 1]
    beam_score[:, :] = prob[:, :, 0]
    for i in range(1, target_len):
        # print(i)
        probs = []
        for j in range(beam_size):
            # batch_size,k,2
            prob = decode_step(model, encode_input, output_topk[:, j, :i + 1], candidates=candidates, beam_size=beam_size)
            probs.append(prob)
        # batch_size,k,k,2
        probs = np.array(probs).swapaxes(0, 1)
        # batch_size,k,k
        beam_scores = np.einsum('abc,ab->abc', probs[:, :, :, 0], beam_score)
        # batch_size,k,2
        top_k_fn = lambda x: np.dstack(np.unravel_index(np.argsort(-x.ravel()), (beam_size, beam_size)))
        top_k_index = np.array(list(map(top_k_fn, beam_scores)))[:, 0][:, :beam_size, :]
        for ii in range(batch_size):
            output_topk[ii, :, :] = output_topk[ii, top_k_index[ii, :, 0], :]
            output_topk[ii, :, i + 1] = probs[ii, top_k_index[ii, :, 0], top_k_index[ii, :, 1], 1]
            beam_score[ii, :] = beam_scores[ii, top_k_index[ii, :, 0], top_k_index[ii, :, 1]]
    return output_topk, beam_score
