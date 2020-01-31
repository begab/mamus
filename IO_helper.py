import re
import gzip
import numpy as np

from zipfile import ZipFile

def load_corpus(corpus_file, load_tags=False):
    if corpus_file.endswith('.gz'):
        corpus = []
        with gzip.open(corpus_file, 'r') as f:
            for line in f:
                corpus.append(line.decode("utf-8").split())
    elif corpus_file.endswith('.conllu'):
        corpus = read_conllUD_file(corpus_file, load_tags)
    return corpus

def read_conllUD_file(location, load_tags):
    sentences = []
    tokens = []
    with open(location) as f:
        for l in f:
            if not(l.strip().startswith('#')):
                s = l.split('\t')
                if len(s) == 10 and not('-' in s[0]):
                    if load_tags:
                        tokens.append((s[1], s[3]))
                    else:
                        tokens.append(s[1])
                elif len(l.strip())==0 and len(tokens) > 0:
                    sentences.append(tokens)
                    tokens = []
    return enforce_unicode(sentences)

def enforce_unicode(sentences):
    """
    In Python3 we should check for str class instead of unicode according to
    https://stackoverflow.com/questions/19877306/nameerror-global-name-unicode-is-not-defined-in-python-3
    """
    if len(sentences) == 0 or type(sentences[0][0][0]) == str: # if the first token is already unicode, there seems nothing to be done
        return sentences
    return [[(unicode(t[0], "utf8"), unicode(t[1], "utf8")) for t in s] for s in sentences]

def load_embeddings(filename, max_words=-1):
    if filename.endswith('.gz'):
        lines = gzip.open(filename)
    elif filename.endswith('.zip'):
        myzip = ZipFile(filename) # we assume only one embedding file to be included in a zip file
        lines = myzip.open(myzip.namelist()[0])
    else:
        lines = open(filename)
    data, words = [], []
    for counter, line in enumerate(lines):
        if len(words) == max_words:
            break
        if type(line) == bytes:
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                print('Error at line {}: {}'.format(counter, line))
                continue
        tokens = line.rstrip().split(' ')
        if len(words) == 0 and len(tokens) == 2 and re.match('[1-9][0-9]*', tokens[0]):
            # the first line might contain the number of embeddings and dimensionality of the vectors
            continue
        try:
            values = [float(i) for i in tokens[1:]]
            if sum([v**2 for v in values])  > 0: # only embeddings with non-zero norm are kept
                data.append(values)
                words.append(tokens[0])
        except:
            print('Error while parsing input line #{}: {}'.format(counter, line))
    i2w = dict(enumerate(words))
    return np.array(data), {v:k for k,v in i2w.items()}, i2w
