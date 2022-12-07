import numpy as np
import joblib
from scipy.sparse import csr_matrix

from fly.vectorizer import vectorize_scale, vectorize
from fly.fly import Fly
from fly.utils import read_vocab
from fly.hash import wta_vectorized
from fly.fly_utils import hash_input_vectorized_


def sort_nonzeros(x):
    sidx = np.argsort(x)
    out_idx = sidx[np.in1d(sidx, np.flatnonzero(x!=0))][::-1]
    out_x = x[out_idx]
    return out_x, out_idx


def return_keywords(vec):
    keywords = []
    vs = np.argsort(vec)
    for i in vs[-10:]:
        if vec[i]:
            keywords.append(i)
    return keywords


def inspect_projection(dataset_name, spf, fly_path=None, logprob_power=1):
    if fly_path:
        # load exist fly
        fly = joblib.load(fly_path)
    else:
        # generate new fly
        pn_size = 10000
        kc_size = 20
        top_words = 10000
        wta_percent = 30
        proj_size = 8
        init_method = "random"
        eval_method = "similarity"
        proj_store = None
        hyperparameters = {'C': 100, 'num_iter': 200, 'num_nns': 10}
        fly = Fly(pn_size, kc_size, wta_percent, proj_size, top_words, init_method, eval_method,
                  proj_store, hyperparameters)

    # read and encode text data
    XX, _, labels = vectorize(dataset_name, spf, logprob_power, fly.top_words)

    # read vocab file
    spm_vocab = f"./spm/{dataset_name}/spm.{dataset_name}.vocab"
    vocab, reverse_vocab, logprobs = read_vocab(spm_vocab)

    # hash the input data
    hashed_kenyon, kc_use, kc_sorted_ids = hash_input_vectorized_(
        pn_mat=csr_matrix(XX),
        weight_mat=fly.projections,
        percent_hash=fly.wta)

    for i in range(XX.shape[0]):
        # print doc idx
        print('###########################')
        print('--- doc idx:', i)

        # display a few important words
        kwds = [reverse_vocab[w] for w in return_keywords(XX[i])]
        print('--- top 10 word: ', kwds)

        # print the projection
        values, ids = sort_nonzeros(hashed_kenyon[i].toarray()[0])
        print('--- num activations: ', len(ids))
        for j in ids:
            words, logprob = [], []
            pn_ids = fly.projections[j].nonzero()[1]
            for k in pn_ids:
                words.append(reverse_vocab[k])
                logprob.append(XX[i][k])
            print('   proj', j, words, logprob)
