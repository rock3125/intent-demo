#!/usr/bin/python3

import os
import sys
import numpy as np
import time
import nltk
import logging
# use natural language toolkit
from nltk.stem.lancaster import LancasterStemmer
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
# import keras.callbacks
from shutil import copyfile


# setup basic logging to console for this system
root = logging.getLogger()
root.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
root.addHandler(ch)


# setup the lancaster stemmer from NLTK
nltk.download('punkt')
stemmer = LancasterStemmer()


# remove all files and sub directories
def clean_up(path):
    if os.path.isdir(path):
        for item in os.listdir(path):
            filename = os.path.join(path, item)
            if os.path.isdir(filename):
                clean_up(filename)
            elif os.path.isfile(filename):
                os.remove(filename)
        if os.path.isdir(path):
            os.removedirs(path)


# tokenize a sentence and stem the words
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


# load a previously saved model and return that model
def load_model(model_json_filename, model_h5_filename, v2i_filename, f2i_filename):
    # load json and create model
    with open(model_json_filename, 'r') as rdr:
        loaded_model_json = rdr.read()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(model_h5_filename)

    # load f2i and v2i
    v2i = []
    v2iSet = dict()
    if os.path.isfile(v2i_filename):
        with open(v2i_filename, 'rt') as vocab:
            for line in vocab:
                line = line.strip()
                if len(line) > 0:
                    v2i.append(line)
                    v2iSet[line] = len(v2iSet)

    f2i = []
    f2iSet = dict()
    if os.path.isfile(f2i_filename):
        with open(f2i_filename, 'rt') as vocab:
            for line in vocab:
                line = line.strip()
                if len(line) > 0:
                    f2i.append(line)
                    f2iSet[line] = len(f2iSet)

    return loaded_model, v2iSet, f2iSet


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, v2iSet):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(v2iSet)
    counter = 0
    for s in sentence_words:
        if s in v2iSet:
            bag[v2iSet[s]] = 1
    return np.asarray(bag)


# perform a prediction using the model and a language sntence
def predict(model, sentence, v2iSet):
    x = bow(sentence.lower(), v2iSet)
    # input layer is our bag of words
    preds = model.predict(np.asarray([x]), verbose=0)[0]
    best_index = -1
    best_value = 0.0
    for i in range(0, len(preds)):
        if preds[i] > best_value:
            best_value = preds[i]
            best_index = i
    return best_index


# convert an index from the outputs to a class name
def prediction_to_class(index, f2iSet):
    for fn, i in f2iSet.items():
        if index == i:
            return fn
    return "?"


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.realpath(__file__))

    model, v2i, f2i = load_model('data/intent_network.model.json', 'data/intent_network.model.h5',
                                 'data/intent_network.v2i', 'data/intent_network.f2i')

    index = predict(model, 'Who is Richard?', v2i)
    print(prediction_to_class(index, f2i))
