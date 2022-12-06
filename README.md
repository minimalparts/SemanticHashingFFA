# Semantic hashing with the Fruit Fly Algorithm

This repository was developed for the paper **Algorithmic Diversity and Tiny Models:
Comparing Binary Neural Networks and the Fruit Fly Algorithm on Document Representation Tasks**.

The paper was presented at SustaiNLP workshop, a venue dedicated for small, efficient 
and environmental friendly models for natural language processing. The workshop was
organized along with EMNLP conference 2022. The paper is part of the [PeARS project](https://pearsproject.org/).

We acknowledge [NLnet](https://nlnet.nl/) for financial support. 

## Install

We recommend creating a virtual environment with Python 3.6 or above. Clone our code:

    git clone https://github.com/PeARSearch/PeARS-fruit-fly.git
    cd PeARS-fruit-fly/

Install requirements:

    pip install -r requirements.txt

## General structures

The Fruit Fly Algorithm is implemented in directory *fly*, while the Binary Neural Network
is located in directory *baseline*. The Binary Neural Network is adapted from
the [original version](https://github.com/itayhubara/BinaryNet.pytorch).

## How to run
The script *run.py* controls all the experiments.

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

The performances are recorded under the directory *baseline* and *fly*, while the
energy consumptions are in the directory *emission_tracking*.

Feel free to contact us if you have difficulties in reproducing the results.
