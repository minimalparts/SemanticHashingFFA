import numpy as np
import argparse
import multiprocessing
from os.path import join
from pathlib import Path
import joblib
from joblib import Parallel, delayed
from sklearn.metrics import pairwise_distances
from collections import Counter
from scipy.spatial.distance import cdist

from fly.classify import train_model
from fly.vectorizer import vectorize


class LSHRandProj:
    def __init__(self, input_dim, hash_dim=32, random_seed=None, eval_method=None, hyperparameters=None):
        self.input_dim = input_dim
        self.hash_dim = hash_dim
        self.random_seed = random_seed
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        self.projections = self.create_projections()
        self.val_score = 0
        self.is_evaluated = False
        # print("INIT",self.kc_size,self.proj_size,self.wta,self.get_coverage())

    def create_projections(self):
        rng = np.random.default_rng(seed=self.random_seed)
        weight_mat = rng.standard_normal(size=(self.input_dim, self.hash_dim))
        return weight_mat

    def evaluate(self, train_set, val_set, train_label, val_label, multiclass, training):
        hash_val = val_set.dot(self.projections)
        hash_val = (hash_val > 0).astype(np.int_)
        if self.eval_method == "classification":
            # We only need the train set for classification, not similarity
            hash_train = train_set.dot(self.projections)
            hash_train = (hash_train > 0).astype(np.int_)
            self.val_score, _ = train_model(m_train=hash_train, classes_train=train_label,
                                            m_val=hash_val, classes_val=val_label,
                                            C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'],
                                            multiclass=multiclass)
        if self.eval_method == "similarity":
            self.val_score = self.prec_at_k(m=hash_val, classes=val_label, k=self.hyperparameters['num_nns'],training=training)
        self.is_evaluated = True
        # print("\nCOVERAGE:",self.get_coverage())
        print("SCORE:", self.val_score)
        # print("PROJECTIONS:",self.print_projections())
        return self.val_score, hash_val

    def compute_nearest_neighbours(self, vecs, labels, i, num_nns):
        i_sim = np.array(vecs[i])
        i_label = labels[i]
        ranking = np.argsort(-i_sim)
        n_labs = [labels[n] for n in ranking][1:num_nns + 1]  # don't count first neighbour which is itself
        n_sum = 0
        if isinstance(n_labs[0], list) == True:
            for lbls in n_labs:
                overlap = len(list(set(lbls) & set(i_label))) / len(i_label)
                n_sum += overlap
        else:
            for lbl in n_labs:
                if lbl == i_label:
                    n_sum += 1
        score = n_sum / num_nns
        return score, n_labs

    def prec_at_k(self, m=None, classes=None, k=None, metric="hamming", training=True):
        if training:
            m = m[:10000]  # Limit eval to 10000 docs
            classes = classes[:10000]
        else:
            m = m[:30000]

        vecs = 1 - pairwise_distances(m, metric=metric)
        scores = []
        for i in range(vecs.shape[0]):
            score, neighbours = self.compute_nearest_neighbours(vecs, classes, i, k)
            scores.append(score)
        return np.mean(scores)


def run_lsh(data=None, spf=None, logprob_power=None, top_words=None, num_trials=None, hash_dim=None, k=None):
    print('--- Running LSH Random Projection ---')
    train_mat, _, labels = vectorize(data, spf, logprob_power, top_words)
    # labels = [l[0] for l in labels]
    model_dir = join(Path(__file__).parent.resolve(), join("models/lsh", data))
    Path(model_dir).mkdir(exist_ok=True, parents=True)
    max_thread = int(multiprocessing.cpu_count() * 1.0)
    # if umap:
    #     train_mat = joblib.load(spf.replace('.sp', '.umap.m'))
    # else:
    #     train_mat = dataset
    train_mat = train_mat  # TODO
    pn_size = train_mat.shape[1]
    eval_method = "similarity"  # TODO
    # eval_method = "classification"
    hyperparameters = {'C': 100, 'num_iter': 200, 'num_nns': k}
    if data in ['reuters', 'tmc']:
        multilabel = True
    else:
        multilabel = False
        labels = [cls[0] for cls in labels]
    lsh_path = join(model_dir, spf.split('/')[-1].replace('sp', str(hash_dim) + '.lsh.m'))

    best_overall_score = 0.0
    print("\n\n----- Initialising", num_trials, "projections ----")
    lsh_list = [LSHRandProj(input_dim=pn_size, hash_dim=hash_dim, random_seed=seed,
                            eval_method=eval_method, hyperparameters=hyperparameters) for seed in range(num_trials)]

    # '''Compute precision at k'''
    print("\n----- Evaluating", num_trials, "lsh ----")
    with Parallel(n_jobs=max_thread, prefer="threads") as parallel:
        delayed_funcs = [delayed(lambda x: x.evaluate(train_mat, train_mat, labels, labels, multilabel, False))(lsh)
                         for lsh in lsh_list]
        score_list = parallel(delayed_funcs)
    # print(score_list)
    scores = np.array([p[0] for p in score_list])
    print("\n\n----- Outputting score list for", num_trials, "lsh ----")
    print(scores)
    best = np.argmax(scores)
    print("BEST:", scores[best])
    if scores[best] > best_overall_score:
        joblib.dump(lsh_list[best], lsh_path)
        best_overall_score = scores[best]
    print("BEST OVERALL FOR HASH DIM", hash_dim, best_overall_score)
    return lsh_path, best_overall_score, scores




