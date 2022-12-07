"""Process a whole Wikipedia dump with the fruit fly

Usage:
  run.py pipeline --dataset=<name>
  run.py train_umap --dataset=<name>
  run.py train_pca --dataset=<name>
  run.py cluster_data --dataset=<name>
  run.py train_fly --dataset=<name> [--raw]
  run.py evaluate --dataset=<name> [--raw]
  run.py analysis --dataset=<name>
  run.py lsh --dataset=<name>

  run.py (-h | --help)
  run.py --version

Options:
  --dataset=<name>             Name of dataset to process.
  pipeline                     Run the whole pipeline (this can take several hours!)
  train_umap                   Train the UMAP dimensionality reduction model.
  train_pca                    Train a PCA dimensionality reduction model (alternative to UMAP).
  cluster_data                 Learn cluster names and apply clustering to the entire Wikipedia.
  train_fly                    Train the fruit fly over dimensionality-reduced representations.
  evaluate                     Run the fruit fly on val and test.
  analysis                     Print the projections for all documents in the input dataset.
  lsh                          Run Locality sensitive hashing (random projection) as a baseline.
  --raw                        When training/applying fruit fly, use raw word vectors, without PCA/UMAP.
  -h --help                    Show this screen.
  --version                    Show version.

"""

import configparser
from os.path import exists
from docopt import docopt
import numpy as np
from random import shuffle
import joblib
import pathlib

from codecarbon import EmissionsTracker
from fly.train_models import train_umap, hack_umap_model, run_pca, hack_pca_model, train_birch, train_fly
from fly.apply_models import apply_dimensionality_reduction, apply_fly
from fly.label_clusters import generate_cluster_labels
from fly.fly_analysis import inspect_projection

from baseline.lsh_rand_proj import run_lsh

def init_config(dataset, is_running_lsh):
    config_path = './configs/' + dataset+'.hyperparameters.cfg'
    if exists(config_path):
        return 1
    else:
        config = configparser.ConfigParser()

        if is_running_lsh:
            config['GENERIC'] = {}
            config['GENERIC']['dataset'] = 'None'
            config['PREPROCESSING'] =  {}
            config['PREPROCESSING']['logprob_power'] = 'None'
            config['PREPROCESSING']['top_words'] =  'None'
            config['REDUCER'] =  {}
            config['REDUCER']['path'] = 'None'
            config['RIDGE'] =  {}
            config['RIDGE']['path'] = 'None'
            config['LSH'] =  {}
            config['LSH']['num_trials'] =  'None'
            config['LSH']['neighbours'] =  'None'
            config['LSH']['32-path'] = 'None'
            config['LSH']['64-path'] = 'None'
            config['LSH']['128-path'] = 'None'
        else:
            config['GENERIC'] = {}
            config['GENERIC']['dataset'] = 'None'
            config['PREPROCESSING'] =  {}
            config['PREPROCESSING']['logprob_power'] = 'None'
            config['PREPROCESSING']['top_words'] =  'None'
            config['REDUCER'] =  {}
            config['REDUCER']['path'] = 'None'
            config['RIDGE'] =  {}
            config['RIDGE']['path'] = 'None'
            config['FLY'] =  {}
            config['FLY']['num_trials'] =  'None'
            config['FLY']['neighbours'] =  'None'
            config['FLY']['32-path'] = 'None'
            config['FLY']['64-path'] = 'None'
            config['FLY']['128-path'] = 'None'
    with open(config_path, 'w+') as configfile:
        config.write(configfile)

def read_config(dataset):
    config_path = './configs/' + dataset+'.hyperparameters.cfg'
    config = configparser.ConfigParser()
    config.read(config_path)
    return config_path, config

def update_config(dataset, section, k, v):
    config_path, config = read_config(dataset)
    config[section][k] = str(v)
    with open(config_path, 'w+') as configfile:
        config.write(configfile)


if __name__ == '__main__':
    args = docopt(__doc__, version='Semantic hashing with the fruit fly, ver 0.1')
    dataset = args['--dataset']

    pathlib.Path(f"./emission_tracking/{dataset}").mkdir(exist_ok=True, parents=True)
    tracker = EmissionsTracker(output_dir=f"./emission_tracking/{dataset}", project_name=f"{dataset} train_lsh")

    init_config(dataset, args['lsh'])

    if dataset == '20news':
        train_path = "./datasets/20news-bydate/20news-bydate-train.sp"
    elif dataset == 'wos':
        train_path = "./datasets/wos/wos11967-train.sp"
    elif dataset == 'wiki':
        train_path = "./datasets/wikipedia/wikipedia-train.sp"
    else:
        train_path = f"./datasets/{dataset}/{dataset}-train.sp"

    
    if args['train_umap'] or args['pipeline']:
        tracker.start()
        umap_path, input_m, umap_m, best_logprob_power, best_top_words = train_umap(dataset, train_path)
        print("UMAP LOG: BEST LOG POWER - ",best_logprob_power, "BEST TOP WORDS:", best_top_words)
        update_config(dataset, 'PREPROCESSING', 'logprob_power', best_logprob_power)
        update_config(dataset, 'PREPROCESSING', 'top_words', best_top_words)
        update_config(dataset, 'REDUCER', 'path', umap_path)
        hacked_path, hacked_m = hack_umap_model(dataset, train_path, best_logprob_power, best_top_words, input_m, umap_m)
        update_config(dataset, 'RIDGE', 'path', hacked_path)
        tracker.stop()

    if args['train_pca']:
        tracker.start()
        pca_path, input_m, pca_m, best_logprob_power, best_top_words = run_pca(dataset, train_path)
        print("PCA LOG: BEST LOG POWER - ",best_logprob_power, "BEST TOP WORDS:", best_top_words)
        update_config(dataset, 'PREPROCESSING', 'logprob_power', best_logprob_power)
        update_config(dataset, 'PREPROCESSING', 'top_words', best_top_words)
        update_config(dataset, 'REDUCER', 'path', pca_path)
        hacked_path, hacked_m = hack_pca_model(dataset, train_path, best_logprob_power, best_top_words, input_m, pca_m)
        update_config(dataset, 'RIDGE', 'path', hacked_path)
        tracker.stop()

    if args['cluster_data'] or args['pipeline']:
        _ , config = read_config(dataset)
        best_logprob_power = int(config['PREPROCESSING']['logprob_power'])
        best_top_words = int(config['PREPROCESSING']['top_words'])
        hacked_path = config['RIDGE']['path']
        hacked_m = joblib.load(hacked_path+'.m') 
        brm, labels = train_birch(dataset, hacked_m)
        generate_cluster_labels(dataset, train_path, labels, best_logprob_power, best_top_words)
        apply_dimensionality_reduction(dataset, train_path, hacked_path, brm, best_logprob_power, best_top_words)

    if args['train_fly'] or args['pipeline']:
        num_trials = 10
        k = 100
        update_config(dataset, 'FLY', 'num_trials', num_trials)
        update_config(dataset, 'FLY', 'neighbours', k)
        _, config = read_config(dataset)
        best_logprob_power = int(config['PREPROCESSING']['logprob_power'])
        best_top_words = int(config['PREPROCESSING']['top_words'])


        if not args['--raw']:
            tracker.start()
        for kc_size in [32, 64, 128]:
            if args['--raw']:
                fly_path, _ = train_fly(dataset, train_path, best_logprob_power, best_top_words, False, num_trials,
                                        kc_size, k)
            else:

                fly_path, _ = train_fly(dataset, train_path, best_logprob_power, best_top_words, True, num_trials,
                                        kc_size, k)
            update_config(dataset, 'FLY', str(kc_size) + '-path', fly_path)
        if not args['--raw']:
            tracker.stop()

    if args['evaluate'] or args['pipeline']:
        _, config = read_config(dataset)
        best_logprob_power = int(config['PREPROCESSING']['logprob_power'])
        best_top_words = int(config['PREPROCESSING']['top_words'])

        for kc_size in [32, 64, 128]:
            print('\n#########')
            print('# ', kc_size, ' #')
            print('#########\n')

            if args['--raw']:
                fly_path = config['FLY'][str(kc_size) + '-path-raw']
                apply_fly(dataset, train_path, fly_path, best_logprob_power, best_top_words, True)
            else:
                fly_path = config['FLY'][str(kc_size) + '-path']
                apply_fly(dataset, train_path, fly_path, best_logprob_power, best_top_words, False)

    if args['analysis']:
        spf = "./datasets/wikipedia/wikipedia_small.sp"
        fly_path = None
        logprob_power = 1
        inspect_projection(dataset_name=dataset, spf=spf,
                           fly_path=fly_path, logprob_power=logprob_power)

    if args['lsh']:
        train_or_test_mode = 'test'  # 'train'
        num_trials = 10  # TODO
        k = 100
        update_config(dataset, 'LSH', 'num_trials', num_trials)
        update_config(dataset, 'LSH', 'neighbours', k)
        _ , config = read_config(dataset)
        best_logprob_power = int(config['PREPROCESSING']['logprob_power'])
        best_top_words = int(config['PREPROCESSING']['top_words'])

        score_list = []
        tracker.start()
        for hash_dim in [32, 64, 128]:
            lsh_path, _, scores = run_lsh(data=dataset, spf=train_path.replace('train', train_or_test_mode),
                                  logprob_power=best_logprob_power, top_words=best_top_words, num_trials=num_trials,
                                  hash_dim=hash_dim, k=k)
            score_list.append(scores)
            update_config(dataset, 'LSH', str(hash_dim)+'-path', lsh_path)
        tracker.stop()
        score_list = np.array(score_list)
        print(score_list, score_list.mean(axis=1))
