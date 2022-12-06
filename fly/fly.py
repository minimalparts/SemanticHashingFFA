"""The main Fly class"""

import gc
import random
from time import time, sleep
from collections import Counter
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse import hstack, vstack, lil_matrix, coo_matrix
from sklearn.metrics import pairwise_distances

from fly.classify import train_model
from fly.fly_utils import read_vocab, hash_dataset_

class Fly:
    def __init__(self, pn_size=None, kc_size=None, wta=None, proj_size=None, top_words=None, init_method=None, eval_method=None, proj_store=None, hyperparameters=None):
        self.pn_size = pn_size
        self.kc_size = kc_size
        self.wta = wta
        self.proj_size = proj_size
        self.top_words = top_words
        self.init_method = init_method
        self.eval_method = eval_method
        self.hyperparameters = hyperparameters
        if self.init_method == "random":
            weight_mat, self.shuffled_idx = self.create_projections(self.proj_size)
        else:
            weight_mat, self.shuffled_idx = self.projection_store(proj_store)

        self.projections = lil_matrix(weight_mat)
        self.val_score = 0
        self.is_evaluated = False
        #print("INIT",self.kc_size,self.proj_size,self.wta,self.get_coverage())

    def create_projections(self,proj_size):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        idx = list(range(self.pn_size))
        random.shuffle(idx)
        used_idx = idx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(0,len(idx),proj_size):
                p = idx[i:i+proj_size]
                for j in p:
                    weight_mat[c][j] = 1
                c+=1
                if c >= self.kc_size:
                    break
            random.shuffle(idx) #reshuffle if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def grow(self, num_new_rows):
        new_mat = np.zeros((num_new_rows, self.pn_size))
        for i in range(num_new_rows):
            if self.init_method == "random":
                for j in np.random.randint(self.pn_size, size=self.proj_size):
                    new_mat[i, j] = 1
            else:
                random.shuffle(self.proj_store)
                p = self.proj_store[0]
                for j in p:
                    new_mat[i][j] = 1
        # concat the old part with the new part
        self.projections = vstack([self.projections, lil_matrix(new_mat)])
        self.projections = lil_matrix(self.projections)
        self.kc_size+=num_new_rows
        return self.kc_size

    def projection_store(self,proj_store):
        weight_mat = np.zeros((self.kc_size, self.pn_size))
        self.proj_store = proj_store.copy()
        proj_size = len(self.proj_store[0])
        random.shuffle(self.proj_store)
        sidx = [pn for p in self.proj_store for pn in p]
        idx = list(range(self.pn_size))
        not_in_store_idx = list(set(idx) - set(sidx))
        #print(len(not_in_store_idx),'IDs not in store')
        used_idx = sidx.copy()
        c = 0

        while c < self.kc_size:
            for i in range(len(self.proj_store)):
                p = self.proj_store[i]
                for j in p:
                    weight_mat[c][j] = 1
                c+=1
                if c >= self.kc_size:
                    break
            random.shuffle(idx) #add random if needed -- if all KCs are not filled
            used_idx.extend(idx)
        return weight_mat, used_idx[:self.kc_size * proj_size]

    def get_coverage(self):
        ps = self.projections.toarray()
        vocab_cov = (self.pn_size - np.where(~ps.any(axis=0))[0].shape[0]) / self.pn_size
        kc_cov = (self.kc_size - np.where(~ps.any(axis=1))[0].shape[0]) / self.kc_size
        return vocab_cov, kc_cov

    def get_fitness(self):
        if not self.is_evaluated:
            return 0
        if DATASET == "all":
            return np.mean(self.val_scores)
        else:
            return np.sum(self.val_scores)

    def hash(self,train_set,val_set,train_label,val_label):
        hash_val, kc_use_val, kc_sorted_val = hash_dataset_(dataset_mat=val_set, weight_mat=self.projections, percent_hash=self.wta, top_words=self.top_words)
        return hash_val, kc_use_val, kc_sorted_val

    def evaluate(self,train_set,val_set,train_label,val_label,multiclass,training):
        hash_val, kc_use_val, kc_sorted_val = self.hash(train_set,val_set,train_label,val_label)
        if self.eval_method == "classification":
            #We only need the train set for classification, not similarity
            hash_train, kc_use_train, kc_sorted_train = hash_dataset_(dataset_mat=train_set, weight_mat=self.projections,
                                   percent_hash=self.wta, top_words=self.top_words)
            self.val_score, _ = train_model(m_train=hash_train, classes_train=train_label,
                                       m_val=hash_val, classes_val=val_label,
                                       C=self.hyperparameters['C'], num_iter=self.hyperparameters['num_iter'], multiclass=multiclass)
        if self.eval_method == "similarity":
            self.val_score = self.prec_at_k(m=hash_val, classes=val_label, k=self.hyperparameters['num_nns'],training=training)
        self.is_evaluated = True
        #print("\nCOVERAGE:",self.get_coverage())
        print("SCORE:",self.val_score)
        #print("PROJECTIONS:",self.print_projections())
        return self.val_score, hash_val


    def compute_nearest_neighbours(self, vecs, labels, i, num_nns):
        i_sim = np.array(vecs[i])
        i_label = labels[i]
        ranking = np.argsort(-i_sim)
        n_labs = [labels[n] for n in ranking][1:num_nns+1] #don't count first neighbour which is itself
        n_sum=0
        if isinstance(n_labs[0], list)==True:
            for lbls in n_labs:
                overlap = len(list(set(lbls) & set(i_label))) / len(i_label)
                n_sum+=overlap
        else:
            for lbl in n_labs:
                if lbl == i_label:
                    n_sum+=1
        score = n_sum / num_nns
        return score,n_labs

    def prec_at_k(self, m=None, classes=None, k=None, metric="hamming", training=True):
        if training:
            m = m.todense()[:10000] #Limit eval to 10000 docs
            classes = classes[:10000]
        else:
            m = m.todense()[:30000]

        vecs = 1-pairwise_distances(m, metric=metric)
        scores = []
        for i in range(vecs.shape[0]):
            score, neighbours = self.compute_nearest_neighbours(vecs,classes,i,k)
            scores.append(score)
        return np.mean(scores)