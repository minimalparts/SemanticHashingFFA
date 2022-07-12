import torch
import torch.optim as optim
from modules import Model

import argparse
import os
import csv
from utils import extract_feat_from_layer, append_in_tsv, data_loader
from eval import prec_at_k

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_feat_bnn():
    # net
    model = Model(num_labels=args.num_labels,
                  dim_features=args.dim_features,
                  neurons_hl=args.neurons_hl)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = torch.load(f'./models/{args.dataset}/checkpoint_{args.lr}_{args.neurons_hl}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    # layer
    layer_name = 'fc1'

    # extract features in hidden layer
    feats, labels = extract_feat_from_layer(net=model,
                                             layer=layer_name,
                                             dataloader=test_loader,
                                             device=device)
    print(feats.shape)  # expect to be 128 dims
    return feats, labels


def run_prec_at_k():
    feats, labels = extract_feat_bnn()

    if args.dataset == "tmc" or args.dataset=="reuters":
        real_labels=[]
        for label in labels:
            real_labels.append([enu for enu,i in enumerate(label) if i==1])
        labels=real_labels

    print("Prec at k, val:")
    score = prec_at_k(m=feats,classes=labels,k=knn,metric="cosine")
    print(score)
    args.prec_at_k=score

    f_save=f'precatk_scores.tsv'
    if not os.path.exists(f_save):
        append_in_tsv(f_save, list(vars(args).keys()))
    append_in_tsv(f_save, list(vars(args).values()))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Binarized weights')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dataset", type=str, default='wiki')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--neurons_hl", type=int, default=512)
    parser.add_argument("--dim_reduction", type=str)
    args = parser.parse_args()

    tsv_file = open("./models/args_small_models.tsv", "r")
    tsv_reader=csv.reader(tsv_file)
    for enu, row in enumerate(tsv_reader):
        if enu == 0:
            index_cols = {k: v for v, k in enumerate(row)}
            print(index_cols)
            continue
        else:
            args.lr = float(row[index_cols['lr']])
            args.dataset = row[index_cols['dataset']]
            args.dim_reduction=row[index_cols['dim_reduction']]
            if args.dim_reduction != "umap":
                args.neurons_hl = int(row[index_cols['neurons_hl']])
                print("ARGS:",args.dim_reduction, args.lr, args.dataset, args.neurons_hl)

                if args.dataset == "wiki":
                    args.train_path = "../datasets/wikipedia/wikipedia-train.sp"
                    args.spm_model = "../spm/wiki/spm.wiki.model"
                    args.spm_vocab = "../spm/wiki/spm.wiki.vocab"
                    args.logprob_power = 5
                if args.dataset == "20news":
                    args.train_path = "../datasets/20news-bydate/20news-bydate-train.sp"
                    args.spm_model = "../spm/20news/spm.20news.model"
                    args.spm_vocab = "../spm/20news/spm.20news.vocab"
                    args.logprob_power = 4
                if args.dataset == "wos":
                    args.train_path = "../datasets/wos/wos11967-train.sp"
                    args.spm_model = "../spm/wos/spm.wos.model"
                    args.spm_vocab = "../spm/wos/spm.wos.vocab"
                    args.logprob_power = 4
                if args.dataset == "reuters" or args.dataset == "tmc" or args.dataset == "agnews":
                    args.train_path = f"../datasets/{args.dataset}/{args.dataset}-train.sp"
                    args.spm_model = f"../spm/{args.dataset}/spm.{args.dataset}.model"
                    args.spm_vocab = f"../spm/{args.dataset}/spm.{args.dataset}.vocab"
                    args.logprob_power = 3

                knn=100
                print("DATASET: ", args.dataset)
                if os.path.exists(f'./models/{args.dataset}/checkpoint_{args.lr}_{args.neurons_hl}.pt'):
                    train_loader, test_loader, val_loader, args.num_labels, args.dim_features = data_loader(args)
                    run_prec_at_k()
                else:
                    print("File doesn't exist!")
    tsv_file.close()
