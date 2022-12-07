import re
import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.metrics import pairwise_distances
from os.path import exists
import csv
import joblib
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer  #MinMax=(0, 1) and Standard=(-1,1)
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter


def transform_dataset(args):
    # global variables
    sp = spm.SentencePieceProcessor()
    sp.load(args.spm_model)
    vocab, reverse_vocab, logprobs = read_vocab(args.spm_vocab)
    vectorizer = CountVectorizer(vocabulary=vocab, lowercase=True, token_pattern='[^ ]+')

    train_set, train_label = read_n_encode_dataset(args.train_path, vectorizer, logprobs, args.logprob_power,
                                                   args.dataset)
    test_set, test_label = read_n_encode_dataset(args.train_path.replace('train', 'test'), vectorizer, logprobs,
                                                 args.logprob_power, args.dataset)
    val_set, val_label = read_n_encode_dataset(args.train_path.replace('train', 'val'), vectorizer, logprobs,
                                               args.logprob_power, args.dataset)

    if args.dataset != "reuters" and args.dataset != 'tmc':
        dic_labels = {}
        labels = sorted(set(train_label))
        for i in range(len(labels)):
            dic_labels[labels[i]] = i
        train_label = torch.Tensor([dic_labels[i] for i in train_label]).type(torch.LongTensor)
        test_label = torch.Tensor([dic_labels[i] for i in test_label]).type(torch.LongTensor)
        val_label = torch.Tensor([dic_labels[i] for i in val_label]).type(torch.LongTensor)
    else:
        labels = set()
        for split in [train_label, val_label, test_label]:
            for i in split:
                for lab in i:
                    labels.add(lab)
        onehotencoder = MultiLabelBinarizer(classes=sorted(labels))
        train_label = torch.Tensor(onehotencoder.fit_transform(train_label))
        test_label = torch.Tensor(onehotencoder.fit_transform(test_label))
        val_label = torch.Tensor(onehotencoder.fit_transform(val_label))

    scaler = StandardScaler().fit(train_set.todense())
    train_set = scaler.transform(train_set.todense())
    val_set = scaler.transform(val_set.todense())
    test_set = scaler.transform(test_set.todense())
    if args.dim_reduction:
        dim_model = joblib.load(f"../../dense_fruit_fly/models/umap/{args.dataset}.{args.dim_reduction}")
        train_set = dim_model.transform(train_set)
        val_set = dim_model.transform(val_set)
        test_set = dim_model.transform(test_set)
    return train_set, train_label, val_set, val_label, test_set, test_label, len(labels), train_set.shape[1]


def data_loader(args):
    train_set, train_label, val_set, val_label, test_set, test_label, num_labels, dim_features = transform_dataset(args)

    train_set = torch.Tensor(train_set)
    test_set = torch.Tensor(test_set)
    val_set = torch.Tensor(val_set)
    print("train_set shape:", train_set.shape)

    train_dataset = TensorDataset(train_set, train_label)  # create your datset
    test_dataset = TensorDataset(test_set, test_label)
    val_dataset = TensorDataset(val_set, val_label)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, num_labels, dim_features


def extract_feat_from_layer(net, layer, dataloader, device):
    """
    Ref: https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3
    Extract features from a specific hidden layer.
    Args:
        net: network model, loaded from torch.load()
        layer: name of the layer, e.g. 'fc2'
        dataloader: pytorch dataloader from read_dataset function
        device: either 'cpu' or 'cuda'
    Returns: features (matrix with shape num_datapoint x num_dimension) and labels
    """

    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    net._modules[layer].register_forward_hook(get_features('feat'))

    feats, labels = [], []
    # loop through batches
    with torch.no_grad():
        for instance, label in dataloader:
            outputs = net(instance.to(device))
            feats.append(features['feat'].detach().cpu().numpy())
            labels.append(label.numpy())
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    return feats, labels


def append_in_tsv(f, row):
  output_file = open(f, 'a', encoding='utf-8')
  tsv_writer= csv.writer(output_file)
  tsv_writer.writerow(row)
  output_file.close()


def read_vocab(vocab_file):
    c = 0
    vocab = {}
    reverse_vocab = {}
    logprobs = []
    with open(vocab_file) as f:
        for l in f:
            l = l.rstrip('\n')
            wp = l.split('\t')[0]
            logprob = -(float(l.split('\t')[1]))
            #logprob = log(lp + 1.1)
            if wp in vocab or wp == '':
                continue
            vocab[wp] = c
            reverse_vocab[c] = wp
            logprobs.append(logprob)
            c+=1
    return vocab, reverse_vocab, logprobs

def read_projections(proj_size):
    proj_store = []
    with open(proj_store_path+str(proj_size)) as f:
        for l in f:
            ps = l.split(" :: ")[0]
            ps = [int(i) for i in ps.split()]
            proj_store.append(ps)
    return proj_store


def projection_vectorized(projection_mat, projection_functions):
    KC_size = len(projection_functions)
    PN_size = projection_mat.shape[1]
    weight_mat = np.zeros((KC_size, PN_size))
    for kc_idx, pn_list in projection_functions.items():
        weight_mat[kc_idx][pn_list] = 1
    weight_mat = coo_matrix(weight_mat.T)
    return projection_mat.dot(weight_mat)


def wta_vectorized(feature_mat, k):
    # thanks https://stackoverflow.com/a/59405060
    m, n = feature_mat.shape
    k = int(k * n / 100)
    # get (unsorted) indices of top-k values
    topk_indices = np.argpartition(feature_mat, -k, axis=1)[:, -k:]
    # get k-th value
    rows, _ = np.indices((m, k))
    kth_vals = feature_mat[rows, topk_indices].min(axis=1)
    # get boolean mask of values smaller than k-th
    is_smaller_than_kth = feature_mat < kth_vals[:, None]
    # replace mask by 0
    feature_mat[is_smaller_than_kth] = 0
    return feature_mat


def read_n_encode_dataset(path, vectorizer, logprobs, power, dataset_name):
    # read
    doc_list, label_list = [], []
    doc = ""
    with open(path) as f:
        for l in f:
            l = l.rstrip('\n')
            if l[:4] == "<doc":
                m = re.search(".*class=([^ ]*)>", l)
                label = m.group(1)
                if dataset_name == 'tmc' or dataset_name == 'reuters':
                    label = [i for i in label.split("|")]
                label_list.append(label)
            elif l[:5] == "</doc":
                doc_list.append(doc)
                doc = ""
            else:
                doc += l + ' '

    # encode
    logprobs = np.array([logprob ** power for logprob in logprobs])
    X = vectorizer.fit_transform(doc_list)
    X = csr_matrix(X)
    X = X.multiply(logprobs)

    return X, label_list


def write_as_json(dic, f):
    output_file = open(f, 'w', encoding='utf-8')
    json.dump(dic, output_file)


def append_as_json(dic, f):
    output_file = open(f, 'a', encoding='utf-8')
    json.dump(dic, output_file)
    output_file.write("\n")


def hash_input_vectorized_(pn_mat, weight_mat, percent_hash):
    kc_mat = pn_mat.dot(weight_mat.T)
    #print(pn_mat.shape,weight_mat.shape,kc_mat.shape)
    kc_use = np.squeeze(kc_mat.toarray().sum(axis=0,keepdims=1))
    kc_use = kc_use / sum(kc_use)
    kc_sorted_ids = np.argsort(kc_use)[:-kc_use.shape[0]-1:-1] #Give sorted list from most to least used KCs
    m, n = kc_mat.shape
    wta_csr = csr_matrix(np.zeros(n))
    for i in range(0, m, 2000):
        part = wta_vectorized(kc_mat[i: i+2000].toarray(), k=percent_hash)
        wta_csr = vstack([wta_csr, csr_matrix(part, shape=part.shape)])
    hashed_kenyon = wta_csr[1:]
    return hashed_kenyon, kc_use, kc_sorted_ids


def hash_dataset_(dataset_mat, weight_mat, percent_hash):
    m, n = dataset_mat.shape
    dataset_mat = csr_matrix(dataset_mat)
    hs, kc_use, kc_sorted_ids = hash_input_vectorized_(dataset_mat, weight_mat, percent_hash)
    hs = (hs > 0).astype(np.int_)
    return hs, kc_use, kc_sorted_ids


