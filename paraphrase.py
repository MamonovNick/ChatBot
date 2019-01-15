from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
import nltk


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


def tag(sentence):
    words = word_tokenize(sentence)
    words = pos_tag(words)
    return words


def paraphraseable(tag):
    return  tag == 'VB' or tag.startswith('JJ')


def synonyms(word, tag):
    lemma_lists = [ss.lemmas() for ss in wn.synsets(word)]
    lemmas = [lemma.name() for lemma in sum(lemma_lists, [])]
    return set(lemmas)


def synonymIfExists(sentence):
    for (word, t) in tag(sentence):
        if paraphraseable(t):
            syns = synonyms(word, t)
            if syns:
                if len(syns) > 1:
                    yield [word, list(syns)]
                    continue
    yield [word, []]


def paraphrase(sentence):
    sent = ""
    for x in synonymIfExists(sentence):
        if not x[1]:
            sent += x[0] + ' '
        else:
            sent += x[1][0] + ' '
    return sent
