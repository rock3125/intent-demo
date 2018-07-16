#!/usr/bin/python3

import os
import sys
import numpy as np
import time
import logging
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import model_from_json
# import keras.callbacks
from shutil import copyfile
from parser.nl_parser import parse_text

# aux verbs - these are verbs you should avoid as they are commonly used as auxiliary verbs in English
avoid_verb_lemmas = {'be', 'have', 'do'}

# tokens to ignore
ignore_words = {'?', '!', '.', '<', '>', ',', ' ', '"', '\'', '(', ')'}

# setup basic logging to console for this system
root = logging.getLogger()
root.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
root.addHandler(ch)


# load the neural network created by Keras (net_model_filename)
# and save it as a java compatible graph in a new clean directory out_dir (must be empty)
def convert_model_to_tf(net_model_filename: str, out_dir: str):
    from keras import backend as K
    import tensorflow as tf
    from tensorflow.python.saved_model import builder as saved_model_builder
    from keras.models import load_model

    try:
        net_model = load_model(net_model_filename)
    except ValueError as err:
        logging.info('''Input file specified ({}) only holds the weights, and not the model defenition.
        Save the model using mode.save(filename.h5) which will contain the network architecture
        as well as its weights. 
        If the model is saved using model.save_weights(filename.h5), the model architecture is 
        expected to be saved separately in a json format and loaded prior to loading the weights.
        Check the keras documentation for more details (https://keras.io/getting-started/faq/)'''
              .format(net_model_filename))
        raise err

    session = K.get_session()

    input_tensor = net_model.inputs[0]
    output_tensor = net_model.outputs[0]

    signature = tf.saved_model.signature_def_utils.build_signature_def(
                        inputs = {'input': tf.saved_model.utils.build_tensor_info(input_tensor)},
                        outputs = {'output': tf.saved_model.utils.build_tensor_info(output_tensor)},)

    b = saved_model_builder.SavedModelBuilder(out_dir)
    b.add_meta_graph_and_variables(session,
                                 [tf.saved_model.tag_constants.SERVING],
                                 signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    b.save()

    # done
    # session.close()

    if net_model_filename.endswith('_full_model.h5'):
        copyfile(net_model_filename.replace('_full_model.h5', '.f2i'), os.path.join(out_dir, 'saved_model.f2i'))
        copyfile(net_model_filename.replace('_full_model.h5', '.v2i'), os.path.join(out_dir, 'saved_model.v2i'))


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
    sentence_list = parse_text(sentence)
    sentence_words = []
    for spacy_sentence in sentence_list:
        sentence_words.extend(spacy_sentence)

    # stem each word
    sentence_words = [word['lemma'] for word in sentence_words if word['tag'].startswith("NN") or word['tag'].startswith("VB")]
    return sentence_words


# save the model to file
def save_model(the_model, filename, v_words, v_classes):
    logging.info('saving model ' + filename)

    # serialize model to JSON
    model_json = the_model.to_json()
    with open(filename + ".model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    the_model.save_weights(filename + ".model.h5")

    # save weight and architecture of the model in this file
    the_model.save(filename + "_full_model.h5")

    # save words array
    with open(filename + '.v2i', 'wt') as writer:
        for word in v_words:
            writer.write(word + '\n')

    # save classes array
    with open(filename + '.f2i', 'wt') as writer:
        for word in v_classes:
            writer.write(word + '\n')


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
    if os.path.isfile(v2i_filename):
        with open(v2i_filename, 'rt') as vocab:
            v2i = json.loads(vocab.read().strip())

    f2i = []
    if os.path.isfile(f2i_filename):
        with open(f2i_filename, 'rt') as vocab:
            f2i = json.loads(vocab.read().strip())

    return loaded_model, v2i, f2i


# round a number to only 3 decimal places
def round_num(n):
    return int(n * 1000.0) / 1000.0


# train a model from scratch until we reach a certain level of certainty or a number of epochs
def _train(model_name, X, y, x_test, y_test, classes, words, max_epochs=100, max_loss=0.01, min_accuracy=0.99,
           hidden_neurons=10, dropout=False, dropout_percent=0.5):

    logging.info("Training with %s neurons, dropout:%s %s" % (hidden_neurons, dropout, dropout_percent if dropout else ''))
    logging.info("Input matrix: %sx%s    Output matrix: %sx%s" % (len(X), len(X[0]), 1, len(classes)))

    if os.path.isfile(model_name + ".model.json"):
        logging.info('re-loading previously saved model')
        model, words, classes = load_model(model_name + '.model.json', model_name + '.model.h5',
                                           model_name + '.v2i', model_name + '.f2i')
    else:
        model = Sequential()
        model.add(Dense(hidden_neurons, input_dim=len(X[0]), activation='sigmoid'))
        if dropout:
            model.add(Dropout(dropout_percent))
        model.add(Dense(len(classes), input_shape=(hidden_neurons,), activation='softmax'))

    # always compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # tensorboard use: tensorboard --logdir ./graph
    # tb_call_back = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=1, write_graph=True,
    #                                            write_images=False)

    # Training a model
    loss = max_loss
    acc = 0.0
    epoch = 0
    while (loss >= max_loss or acc < min_accuracy) and epoch < max_epochs:
        h = model.fit(X, y, epochs=1, verbose=0,
                      batch_size=10, validation_data=(x_test, y_test))  # , callbacks=[tb_call_back])
        loss = h.history['loss'][-1]
        acc = h.history['acc'][-1]
        epoch += 1
        logging.info('epoch ' + str(epoch) + ", loss " + str(round_num(loss)) + ", acc " + str(round_num(acc)))

    return model


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0] * len(words)
    counter = 0
    for s in sentence_words:
        if s not in ignore_words:
            found = False
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
                    found = True
                    break
            if found:
                counter += 1
    return np.asarray(bag), counter


# perform a prediction using the model and a language sntence
def predict(model, sentence, words):
    x, num_words = bow(sentence.lower(), words)
    if num_words == 0:  # couldn't find any words
        return []
    # input layer is our bag of words
    return model.predict(np.asarray([x]), verbose=0)[0]


# convert an index from the outputs to a class name
def prediction_to_class(index, classes):
    if 0 <= index <= len(classes):
        return classes[index]
    return "?"


# go through the loaded training data and remove common words that exist cross classes
# also - remove any training examples already seen, and check for empty classes at the end
# returns the updated training_data list
def remove_duplicates_from_training_data(training_data):
    # adjust the training data for the classes - removing words used by both classes
    # 1. collect the words used by all classes
    class_dictionary = dict()
    for item in training_data:
        curr_class = item['class']
        if curr_class not in class_dictionary:
            class_dictionary[curr_class] = set()
        for word in item['sentence']:
            class_dictionary[curr_class].add(word)

    # 2. remove words from training data
    new_training_data = []
    training_data_seen = set()
    class_counter = dict()
    for item in training_data:
        curr_class = item['class']
        curr_words = item['sentence']

        # count members
        if curr_class not in class_counter:
            class_counter[curr_class] = 0

        new_words = []
        for word in curr_words:
            found = False
            for d_class, d_words in class_dictionary.items():
                if d_class != curr_class and not found:
                    if word in d_words:
                        found = True
            if not found:
                new_words.append(word)
        if len(new_words) > 0:
            if new_words.__str__() not in training_data_seen:
                training_data_seen.add(new_words.__str__())
                data = {"class": curr_class, "sentence": new_words, "text": item['text']}
                new_training_data.append(data)
                class_counter[curr_class] += 1

    # check for empty classes - very bad!
    for class_name, count in class_counter.items():
        if count == 0:
            raise ValueError('class ' + class_name + ' is empty / has no unique words')

    return new_training_data


# start a training session
def train(training_data_filename, max_epochs):

    base_dir = os.path.dirname(os.path.realpath(__file__))

    # check parameter exists and is a file
    if not os.path.isfile(training_data_filename):
        raise ValueError("training file not found: " + training_data_filename)

    # set the stage for progress
    logging.info("epoch 0, loss 10.0, acc 0.0")

    # 3 classes of training data
    training_data = []

    # work out frequencies of training data items
    class_frequencies = dict()
    with open(training_data_filename) as reader:
        for line in reader:
            line = line.strip()
            parts = line.split("::")
            if len(parts) == 2:
                if parts[0] not in class_frequencies:
                    class_frequencies[parts[0]] = [parts[1]]
                else:
                    class_frequencies[parts[0]].append(parts[1])

    # get the size of the largest sample
    max_class_size = 0
    for v in class_frequencies.values():
        max_class_size = max(max_class_size, len(v))

    # setup classes and vocab according to largest sample
    words = []

    classes = []
    class_set = set()

    nnet_vocab = []
    vocab_set = set()

    for cat, list in class_frequencies.items():
        sentence_cache = dict()
        for index in range(0, max_class_size):
            sentence = list[index % len(list)]

            if not sentence in sentence_cache:
                # tokenize in each word in the sentence
                token_list = clean_up_sentence(sentence)

                # add to our classes list
                if cat not in class_set:
                    class_set.add(cat)
                    classes.append(cat)

                # create a properly tokenized sentence
                new_sentence = []
                i = 0
                while i < len(token_list):
                    w1 = token_list[i]
                    if w1 == "<" and i + 2 < len(token_list):
                        if token_list[i + 2] == ">":
                            w1 = "<" + token_list[i + 1] + ">"
                            new_sentence.append(w1)
                            i += 3
                        else:
                            new_sentence.append("<")
                            i += 1
                    elif w1 not in ignore_words:
                        new_sentence.append(w1)
                        i += 1
                    else:
                        i += 1

                data = {"class": cat, "sentence": new_sentence, "text": sentence}
                training_data.append(data)
                sentence_cache[sentence] = data

                # cache vocab set
                for w in new_sentence:
                    if w not in vocab_set:
                        vocab_set.add(w)
                        nnet_vocab.append(w)

            else:
                training_data.append(sentence_cache[sentence])

    # clean data
    training_data = remove_duplicates_from_training_data(training_data)

    # remove the data and log directories
    clean_up(os.path.join(base_dir,'data'))
    clean_up(os.path.join(base_dir,'logdir'))

    # organizing our data structures for documents , classes, words
    if not os.path.isdir(os.path.join(base_dir, "data")):
        os.mkdir(os.path.join(base_dir, "data"))

    # stem and lower each word and remove duplicate
    words = nnet_vocab
    words.sort()

    # remove duplicates
    classes.sort()

    logging.info(str(len(training_data)) + " documents")
    logging.info(str(len(classes)) + " classes")
    logging.info(str(len(words)) + " unique stemmed words")

    # create our training data, strings to numbers
    training = []
    output = []
    test_x = []
    test_y = []

    # training set, bag of words for each sentence
    counter = 0
    td_cache = dict()
    for doc in training_data:
        if doc['text'] not in td_cache:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = set()
            for w in doc['sentence']:
                pattern_words.add(w)
            # create our bag of words array
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            # output is a 0 for each tag and 1 for current tag
            output_row = [0] * len(classes)
            output_row[classes.index(doc['class'])] = 1
            td_cache[doc['text']] = (bag, output_row)
        else:
            bag, output_row = td_cache[doc['text']]

        # 1 in 15 split training and testing
        if counter % 15 == 0:
            test_x.append(bag)
            test_y.append(output_row)

        # still collect all data because we can't afford to lose any
        training.append(bag)
        output.append(output_row)

        counter += 1

    # start training
    m_filename = os.path.join(base_dir, "data/intent_network")
    start_time = time.time()
    X = np.asarray(training)
    y = np.asarray(output)
    model = _train(m_filename, X, y, np.asarray(test_x), np.asarray(test_y), classes, words, max_epochs=max_epochs,
                   hidden_neurons=20, dropout=False, dropout_percent=0)

    # save the model
    save_model(model, m_filename, words, classes)

    # and make it Java compatible
    convert_model_to_tf(os.path.join(base_dir, 'data/intent_network_full_model.h5'), os.path.join(base_dir, 'logdir'))

    elapsed_time = time.time() - start_time
    logging.info("neural network training time: {:.2f} seconds".format(elapsed_time))

    return model, words, classes
