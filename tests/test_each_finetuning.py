"""
    CS635 Final Project - Group 9
    Contributed: Corey Urbanke
    Desc:
    preliminary reproducible results and unit tests for original code base
    done for the Project Proposal presentation

    Modifications: (entire file added)
"""
from __future__ import absolute_import, print_function, division, unicode_literals

import test_helper

from nose.plugins.attrib import attr
import json
import numpy as np

from torchmoji.class_avg_finetuning import relabel
from torchmoji.sentence_tokenizer import SentenceTokenizer

from torchmoji.finetuning import (
    calculate_batchsize_maxlen,
    freeze_layers,
    change_trainable,
    finetune,
    load_benchmark
    )
from torchmoji.model_def import (
    torchmoji_transfer,
    torchmoji_feature_encoding,
    torchmoji_emojis
    )
from torchmoji.global_variables import (
    PRETRAINED_PATH,
    NB_TOKENS,
    VOCAB_PATH,
    ROOT_PATH
    )



@attr('slow')
def test_finetune_last():
    """ finetuning a SS-Twitter model using 'last'.
    """
    dataset_path = ROOT_PATH + '/data/SCv1/raw.pickle'
    nb_classes = 2
    # min_acc = 0.68
    min_acc = .2

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    data = load_benchmark(dataset_path, vocab)
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
    print(model)
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='last', nb_epochs=1) #metric='weighted_f1')

    print("Finetune last SS-Twitter acc: {}".format(acc))

    assert acc >= min_acc

@attr('slow')
def test_finetune_full():
    """ finetuning a SS-Twitter model using 'full'.
    """
    dataset_path = ROOT_PATH + '/data/SCv1/raw.pickle'
    nb_classes = 2
    # min_acc = 0.68
    min_acc = .2

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    data = load_benchmark(dataset_path, vocab)
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
    print(model)
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='full', nb_epochs=1) #metric='weighted_f1')

    print("Finetune full SS-Twitter acc: {}".format(acc))

    assert acc >= min_acc

@attr('slow')
def test_finetune_new():
    """ finetuning a SS-Twitter model using 'new'.
    """
    dataset_path = ROOT_PATH + '/data/SCv1/raw.pickle'
    nb_classes = 2
    # min_acc = 0.68
    min_acc = .2

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    data = load_benchmark(dataset_path, vocab)
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
    print(model)
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='new', nb_epochs=1) #metric='weighted_f1')

    print("Finetune new SS-Twitter acc: {}".format(acc))

    assert acc >= min_acc

@attr('slow')
def test_finetune_chain():
    """ finetuning a SS-Twitter model using 'chain-thaw'.
    """
    dataset_path = ROOT_PATH + '/data/SCv1/raw.pickle'
    nb_classes = 2
    # min_acc = 0.68
    min_acc = .2

    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)

    data = load_benchmark(dataset_path, vocab)
    print('Loading model from {}.'.format(PRETRAINED_PATH))
    model = torchmoji_transfer(nb_classes, PRETRAINED_PATH)
    print(model)
    model, acc = finetune(model, data['texts'], data['labels'], nb_classes,
                          data['batch_size'], method='chain-thaw', nb_epochs=1) #metric='weighted_f1')

    print("Finetune chain-thaw SS-Twitter acc: {}".format(acc))

    assert acc >= min_acc

