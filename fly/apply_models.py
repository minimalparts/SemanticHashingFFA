import joblib
import pickle
from glob import glob
from os.path import join, exists
from sklearn.cluster import Birch
from sklearn.linear_model import Ridge
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from collections import Counter
import numpy as np
import umap
from fly.vectorizer import vectorize_scale, vectorize, scale
from fly.fly import Fly


def apply_umap(lang, umap_model, spf, logprob_power, top_words, save=True):
    print('\n---Applying UMAP---')
    dataset, titles = vectorize_scale(lang, spf, logprob_power, top_words)
    m = csr_matrix(umap_model.transform(dataset[:20000, :]))

    for i in range(20000, dataset.shape[0], 20000):
        print("Reducing", i, "to", i + 20000)
        m2 = csr_matrix(umap_model.transform(dataset[i:i + 20000, :]))
        m = vstack((m, m2))
    dataset = np.nan_to_num(m)

    if save:
        dfile = spf.replace('.sp', '.umap.m')
        joblib.dump(dataset, dfile)


def apply_birch(brc, dataset, data_titles, spf, save=True):
    print('--- Cluster matrix using pretrained Birch ---')
    # Cluster points in matrix m, in batches of 20k
    m = joblib.load(spf.replace('.sp', '.umap.m'))
    idx2clusters = list(brc.predict(m[:20000, :]))
    m = m.todense()

    for i in range(20000, m.shape[0], 20000):
        print("Clustering", i, "to", i + 20000)
        idx2clusters.extend(list(brc.predict(m[i:i + 20000, :])))

    print('--- Save Birch output in idx2cl pickled file ---')
    # Count items in each cluster, using labels for whole data
    cluster_counts = Counter(idx2clusters)
    print(len(idx2clusters), cluster_counts)

    if save:
        pklf = spf.replace('sp', 'idx2cl.pkl')
        with open(pklf, 'wb') as f:
            pickle.dump(idx2clusters, f)


def apply_hacked_umap(data, ridge, spf, logprob_power, top_words, save=True):
    dataset, titles = vectorize_scale(data, spf, logprob_power, top_words)
    m = csr_matrix(ridge.predict(dataset[:20000, :]))

    for i in range(20000, dataset.shape[0], 20000):
        print("Reducing", i, "to", i + 20000)
        m2 = csr_matrix(ridge.predict(dataset[i:i + 20000, :]))
        m = vstack((m, m2))
    dataset = np.nan_to_num(m)

    if save:
        dfile = spf.replace('.sp', '.umap.m')
        joblib.dump(dataset, dfile)
    return dataset, titles


def fly(fly_model=None, train_mat=None, mat=None, train_labels=None, spf_labels=None, multilabel=False):
    print('--- Apply fly ---')
    print(train_mat.shape, len(train_labels), mat.shape, len(spf_labels))
    # Compute precision at k using cluster IDs from Birch model
    print("Precision at k")
    fly_model.eval_method = 'similarity'
    score, hashed_data = fly_model.evaluate(mat, mat, spf_labels, spf_labels, multilabel, False)
    print("Classification")
    fly_model.eval_method = 'classification'
    score, hashed_data = fly_model.evaluate(train_mat, mat, train_labels, spf_labels, multilabel, False)
    return score


def apply_dimensionality_reduction(data, spf_train, hacked_path, birch_model, logprob_power, top_words):
    ridge_model = joblib.load(hacked_path)

    spfs = [spf_train, spf_train.replace('train', 'val'), spf_train.replace('train', 'test')]
    for spf in spfs:
        dataset, titles = apply_hacked_umap(data, ridge_model, spf, logprob_power, top_words, True)
        apply_birch(birch_model, dataset, titles, spf, True)


def apply_fly(data, spf_train, fly_path, logprob_power, top_words, raw):
    spfs = [spf_train, spf_train.replace('train', 'val'), spf_train.replace('train', 'test')]
    data_split_labels = []
    mats = []

    for spf in spfs:
        if raw:
            mat, titles, labels = vectorize(data, spf, logprob_power, top_words)
            mat = scale(mat)
        else:
            train_mat = joblib.load(spf_train.replace('.sp', '.umap.m'))
            mat = joblib.load(spf.replace('.sp', '.umap.m'))
            _, _, labels = vectorize(data, spf, logprob_power, top_words)
        data_split_labels.append(labels)
        mats.append(mat)

    for i in range(len(spfs)):
        print('\n##', spfs[i], '##')

        if data in ['reuters', 'tmc']:
            multilabel = True
        else:
            multilabel = False
            for j in range(len(data_split_labels)):
                new_lbls = [cls[0] for cls in data_split_labels[j]]
                data_split_labels[j] = new_lbls

        print("--- FLY PATH", fly_path, " ---")
        fly_model = joblib.load(fly_path)
        print([len(d) for d in data_split_labels])
        fly(fly_model, mats[0], mats[i], data_split_labels[0], data_split_labels[i], multilabel)