""" 
    CS635 Final Project - Group 9
    Description:
    Perform a grid search on number of biLSTM hidden layers
    Train for 10 epochs, 5 times for each layer number, average them up
    just like the original paper
    Assess accuracy and time taken foreach
    Output results
    @author Corey Urbanke

    Modifications: (entire file added)

"""
from __future__ import print_function
import sys
import numpy as np
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))
import json
import math
from torchmoji.lstm import LSTMHardSigmoid
from torchmoji.model_def import torchmoji_transfer
from torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH
from torchmoji.finetuning import (
     load_benchmark,
     finetune)
import time

def roundup(x):
    return int(math.ceil(x / 10.0)) * 10


# Format: (dataset_name,
#          path_to_dataset,
#          nb_classes)
DATASETS = [
    # ('SE0714', '../data/SE0714/raw.pickle', 3, True),
    #  ('SS-Youtube', '../data/SS-Youtube/raw.pickle', 2),
    ('SCv1', '../data/SCv1/raw.pickle', 2, True),
      ]
p = DATASETS[0]
dset = p[0]
path = p[1]
nb_classes = p[2]
extend_with = 10000

RESULTS_DIR = 'results'
# RESULTS_DIR = ''
# 'new' | 'last' | 'full' | 'chain-thaw'
# for the experiment we will be using chain-thaw for all
FINETUNE_METHOD = 'chain-thaw'
VERBOSE = 1
nb_tokens = 50000
# nb_epochs = 1000
nb_epochs = 5
epoch_size = 1000

# layer counts for experiment: 128, 256, 512, 1024
# USAGE: python cs635experiment.py [layer_count] > [output.txt]
layer_count = int(sys.argv[1])
print("Layer count input: ", layer_count)

# Load our vocabulary
with open(VOCAB_PATH, 'r') as vocab_file:
    vocab = json.load(vocab_file)

f = open('{}/cs635_{}_{}_runs_{}_results.txt'.
                format(RESULTS_DIR, dset, FINETUNE_METHOD, layer_count),
                "w")

f.write("Layer count input: {}".format(layer_count))
# to change this to run the entire experiment at one time, 
# we can add for loop and tab over the entire below code:
# note: takes several hours
# layer_counts = [128, 256, 512, 1024]
# for layer_count in layer_counts:
layer_start_time = time.time()
print(f"Running with {layer_count} LSTM layers...")
f.write(f"Running with {layer_count} LSTM layers...")
avg_accuracy = []
# Run 5 times, and average, similar to how paper does it
for rerun_iter in range(5):
    

    # Load dataset.
    data = load_benchmark(path, vocab, extend_with=extend_with)

    (X_train, y_train) = (data['texts'][0], data['labels'][0])
    (X_val, y_val) = (data['texts'][1], data['labels'][1])
    (X_test, y_test) = (data['texts'][2], data['labels'][2])

    weight_path = PRETRAINED_PATH if FINETUNE_METHOD != 'new' else None
    model = torchmoji_transfer(
                nb_classes,
                weight_path,
                extend_embedding=data['added'], lstm_layer_count=layer_count)
    
    print(model)

    # Training
    print('Training: {}'.format(path))
    start = time.time()
    model, result = finetune(model, data['texts'], data['labels'],
                                nb_classes, data['batch_size'],
                                FINETUNE_METHOD, metric='acc',
                                verbose=VERBOSE, nb_epochs=nb_epochs)
    end = time.time()
    print(f"Run {rerun_iter} training time {end - start}s")
    f.write(f"==== Run {rerun_iter} training time {end - start}s")

    print('Test accuracy (dset = {}): {}'.format(dset, result))
    f.write('Test accuracy for run {} (dset = {}): {}'.format(rerun_iter, dset, result))
    avg_accuracy.append(result)

layer_end_time = time.time()

print("Average accuracy for {} LSTM layers: {}".format(layer_count, np.mean(avg_accuracy)))
f.write("Average accuracy for {} LSTM layers: {}".format(layer_count, np.mean(avg_accuracy)))
print("Time spent on {} layer 5 runs: {}".format(layer_count, layer_end_time - layer_start_time))
f.write("--> Time spent on {} layer 5 runs: {}".format(layer_count, layer_end_time - layer_start_time))


f.close()
