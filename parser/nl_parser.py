import os
import en_core_web_sm
import json
import sys


# use spacy small model, disable ner and statistical parser
nlp = en_core_web_sm.load(disable=['parser', 'ner'])  # ~5x faster without parser and ner
nlp.add_pipe(nlp.create_pipe('sentencizer'))  # needs this instead

# unicode characters that need special attention / replacement
white_space = {'\t',  '\r',  '\n', '\u0008', '\ufeff', '\u303f', '\u3000', '\u2420', '\u2408', '\u202f', '\u205f', '\u2000', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u200b'}
full_stops = {'\u002e', '\u06d4', '\u0701', '\u0702', '\ufe12', '\ufe52', '\uff0e', '\uff61'}
single_quotes = {'\u02bc', '\u055a', '\u07f4', '\u07f5', '\u2019', '\uff07', '\u2018', '\u201a', '\u201b', '\u275b', '\u275c'}
double_quotes = {'\u00ab', '\u00bb', '\u07f4', '\u07f5', '\u2019', '\uff07', '\u201c', '\u201d', '\u201e', '\u201f', '\u2039', '\u203a', '\u275d', '\u276e', '\u2760', '\u276f'}
hypens = {'\u207b', '\u208b', '\ufe63', '\uff0d', '\u2014'}


# cleanup UTF-8 characters that could confuse spaCy
def _cleanup_text(text_binary):
    text = text_binary.decode('UTF-8')
    string_list = []  # python string-builder equivalent
    for ch in text:
        if ch in full_stops:
            string_list.append(".")
        elif ch in single_quotes:
            string_list.append("'")
        elif ch in double_quotes:
            string_list.append('"')
        elif ch in hypens:
            string_list.append("-")
        elif ch in white_space:
            string_list.append(" ")
        else:
            string_list.append(ch)
    return ''.join(string_list)


# parse text using spacy
def parse_text(text):
    # convert from spacy internal to dict
    def _convert_spacy_sentence(sent):
        token_list = []
        for token in sent:
            ancestors = []
            for an in token.ancestors:
                ancestors.append(an.i)
            text = token.text.strip()
            if len(text) > 0:  # keep the tags we want - add others here
                token_list.append({'text': text, 'i': token.i, 'tag': token.tag_, 'lemma': token.lemma_})
        return token_list

    # spaCy magic
    tokens = nlp(text)
    sentence_list = []
    for sent in tokens.sents:
        sentence_list.append(_convert_spacy_sentence(sent))
    return sentence_list


if __name__ == "__main__":

    parse_text('he was glad to see her.')

    if len(sys.argv) != 3:
        print("takes two parameters: /path/to/file.txt /path/to/parsed/output/file.txt")
        exit(0)

    in_file = sys.argv[1]
    if not os.path.isfile(in_file):
        raise ValueError("input file does not exist: " + in_file)

    # parse and create new output csv text file
    out_file = sys.argv[2]
    counter = 0
    with open(out_file, 'wt') as writer:
        with open(in_file, 'rb') as reader:
            text_list = reader.read().decode('utf-8', errors='replace').split('\n')
            total_size = len(text_list)
            for line in text_list:
                sentence_list = json.dumps(parse_text(line))
                writer.write(json.dumps(sentence_list) + '\n')

                if counter % 10_000 == 0:
                    print(str(counter) + ' of ' + str(total_size))

                counter += 1
