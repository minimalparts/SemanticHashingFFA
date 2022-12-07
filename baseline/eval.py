import numpy as np
from sklearn.metrics import pairwise_distances

def compute_nearest_neighbours(vecs,labels,i,num_nns):
    i_sim = np.array(vecs[i])
    i_label = labels[i]
    ranking = np.argsort(-i_sim)
    neighbours = [labels[n] for n in ranking][1:num_nns+1] #don't count first neighbour which is itself
    n_sum=0
    if isinstance(neighbours[0], list)==True:
        for n in neighbours:
            for lab in n:
                if lab in i_label:
                    n_sum+=1
                    break
    else:
        for n in neighbours:
            if n == i_label:
                n_sum+=1
    score = n_sum / num_nns
    # print(i,i_label,neighbours,score)
    return score,neighbours

def prec_at_k(m=None,classes=None,k=None,metric="hamming"):
    vecs = 1-pairwise_distances(m, metric=metric)
    scores = []
    for i in range(vecs.shape[0]):
        score, neighbours = compute_nearest_neighbours(vecs,classes,i,k)
        scores.append(score)
    return np.mean(scores)
