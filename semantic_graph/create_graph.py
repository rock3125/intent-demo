from parser.text_loader import file_based_json_generator

# aux verbs - these are verbs you should avoid as they are commonly used as auxiliary verbs in English
avoid_verb_lemmas = {'be', 'have', 'do'}


# add a relationship between two words and their types
def add_relationship(w1, pos1, w2, pos2, score, frequency_set):
    key1 = w1 + ':' + pos1
    key2 = w2 + ':' + pos2
    if key1 not in frequency_set:
        frequency_set[key1] = {}
    set1 = frequency_set[key1]
    if key2 not in set1:
        set1[key2] = score
    else:
        set1[key2] += score


# is this word a valid ascii word like 'car' or 'Fahrenheit456'
def is_valid_word(word):
    if word in avoid_verb_lemmas:
        return False
    for i in range(0, len(word)):
        ch = word[i]
        if not('0' <= ch <= '9' or 'a' <= ch <= 'z' or 'A' <= ch <= 'Z'):
            return False
    return True


# investigate the window defined by token_list[left:right] for a token at offset
def add_window(lemma, pos, offset, token_list, left, right, window_size, frequency_set):
    for i in range(left, right):
        token = token_list[i]
        tag = token['tag']
        if token['lemma'] != lemma and is_valid_word(token['lemma']):
            if pos == 'n' and tag.startswith('NN'):
                dist = (window_size / abs(i - offset)) / window_size
                add_relationship(lemma, pos, token['lemma'], 'n', dist, frequency_set)
            if pos == 'v' and tag.startswith('VB'):
                dist = (window_size / abs(i - offset)) / window_size
                add_relationship(lemma, pos, token['lemma'], 'v', dist, frequency_set)


# investigate a list of sentences for a given window size
def investigate(sentence_list, frequency_set, window_size=20):
    token_list = []
    for sentence in sentence_list:
        token_list.extend(sentence)

    size = len(token_list)
    for i in range(0, size):
        token = token_list[i]
        # a noun?
        tag = token['tag']
        if tag.startswith('NN') or tag.startswith('VB'):
            pos = 'n'
            if tag.startswith('VB'):
                pos = 'v'
            left = max(0, i - window_size)
            right = min(i + window_size, size)
            if is_valid_word(token['lemma']):
                add_window(token['lemma'], pos, i, token_list, left, right, window_size, frequency_set)


#
# create a semantic graph
#
# in_json_file_list: a string filename or a list of filenames to read / process
# max_related_score: is the best score you can hope for being related to another word this is used in normalization
#                    to get below the 1.0 100% scoring
# window_size: how many words to go left and right of focus words
#
def create_graph(in_json_file_list, graph_output_filename, max_related_score=0.75, window_size=10):

    # the data collection dict
    frequency_set = {}

    # open up the set of files (or file) to read
    iterator = file_based_json_generator(in_json_file_list)
    counter = 0
    for json_value in iterator:
        investigate(json_value, frequency_set, window_size)
        counter += 1
        if counter % 10_000 == 0:
            print(counter / 10_000)

    # these are all the words that are nouns and verbs (their lemmas)
    words = [w for w in frequency_set.keys()]
    words.sort()

    # normalize the frequencies of the words to be [0.0 , max_related_score]
    final_set = {}
    for w in words:
        set1 = frequency_set[w]
        set_words = []
        max_freq = 0.0
        for key, freq in set1.items():
            if freq > max_freq:
                max_freq = freq
        max_freq /= max_related_score  # adjust for maximum allowed
        for key, freq in set1.items():
            set_words.append((key, freq / max_freq))
        set_words.sort()
        final_set[w] = set_words

    # todo: filter journo crap
    # now check the final word set relative to known semantic relationships using word2vec distance measurements

    # save to file
    with open(graph_output_filename, 'wt') as writer:
        for key in final_set.keys():
            word_set = final_set[key]
            parts = [key]
            for word, freq in word_set:
                if freq > 0.001:  # cut-off frequencies that are too rare
                    parts.append(word + ',' + "{0:.4f}".format(freq))
            writer.write('|'.join(parts) + '\n')
